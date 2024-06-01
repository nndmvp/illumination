import numpy as np
import cv2
import os
import torch
from PIL import Image
import cupy as cp
import traceback


def LSH(rgb_img, sigma, nbin):
    # 定义颜色范围
    color_max = 255
    color_range = np.arange(0, color_max + 1, color_max / nbin)

    # 计算 alpha 值
    alpha_x = np.exp(-np.sqrt(2) / (sigma * rgb_img.shape[0]))
    alpha_y = np.exp(-np.sqrt(2) / (sigma * rgb_img.shape[1]))

    def process_channel(channel):
        q_mtx = cp.zeros((rgb_img.shape[0], rgb_img.shape[1], nbin))
        for i in range(nbin):
            tmp_img = cp.array(channel)

            mask_l = tmp_img >= color_range[i]  # 获取大于等于下界的像素
            mask_u = tmp_img < color_range[i + 1]  # 获取小于上界的像素
            mask = cp.logical_and(mask_l, mask_u)  # 找到交集像素
            tmp_img[:] = 0
            tmp_img[mask] = 1
            q_mtx[:, :, i] = tmp_img

        # 初始化 hist_mtx 和 f_mtx
        hist_mtx = q_mtx.copy()
        f_mtx = cp.ones_like(q_mtx)

        # x 维度
        # 计算左部分
        hist_mtx_l = hist_mtx.copy()
        f_mtx_l = f_mtx.copy()
        for i in range(1, hist_mtx.shape[1]):
            hist_mtx_l[:, i, :] += alpha_x * hist_mtx_l[:, i - 1, :]
            f_mtx_l[:, i, :] += alpha_x * f_mtx_l[:, i - 1, :]
        # 计算右部分
        hist_mtx_r = hist_mtx.copy()
        f_mtx_r = f_mtx.copy()
        for i in range(hist_mtx.shape[1] - 2, -1, -1):
            hist_mtx_r[:, i, :] += alpha_x * hist_mtx_r[:, i + 1, :]
            f_mtx_r[:, i, :] += alpha_x * f_mtx_r[:, i + 1, :]
        # 合并右部分和左部分
        hist_mtx = hist_mtx_r + hist_mtx_l - q_mtx
        f_mtx = f_mtx_r + f_mtx_l - 1

        # y 维度
        # 计算左部分
        hist_mtx_l = hist_mtx.copy()
        f_mtx_l = f_mtx.copy()
        for i in range(1, hist_mtx.shape[0]):
            hist_mtx_l[i, :, :] += alpha_y * hist_mtx_l[i - 1, :, :]
            f_mtx_l[i, :, :] += alpha_y * f_mtx_l[i - 1, :, :]
        # 计算右部分
        hist_mtx_r = hist_mtx.copy()
        f_mtx_r = f_mtx.copy()
        for i in range(hist_mtx.shape[0] - 2, -1, -1):
            hist_mtx_r[i, :, :] += alpha_y * hist_mtx_r[i + 1, :, :]
            f_mtx_r[i, :, :] += alpha_y * f_mtx_r[i + 1, :, :]
        # 合并右部分和左部分
        hist_mtx = hist_mtx_r + hist_mtx_l - q_mtx
        f_mtx = f_mtx_r + f_mtx_l - 1
        # 使用归一化因子对 H 进行归一化
        hist_mtx /= f_mtx
        k = 0.033
        step = color_max / nbin
        # 计算 rp1
        rp1 = k * channel

        # 获取当前的 bin 值
        bp = channel // step

        # 计算 b 的范围
        bins = cp.arange(nbin)

        # 计算直方图权重
        # 计算最大值
        max_value = cp.maximum(k, rp1)
        # 扩展 max_value 的维度，使其与 numerator 的形状相匹配
        max_value_expanded = cp.expand_dims(max_value, axis=-1)
        # 计算分子
        numerator = cp.square(cp.expand_dims(bins, axis=(0, 1)) - cp.expand_dims(bp, axis=2))
        # 计算权重
        hist_weights = cp.exp(-numerator / (2 * cp.square(max_value_expanded)))

        # 计算 Ip
        Ip = cp.sum(hist_mtx * hist_weights, axis=2)

        # 将 Ip 赋值给输出图像
        img_out = Ip
        # 将图像像素归一化到 [0, 255] 范围内
        img_out_normalized = cp.maximum(0, cp.minimum(img_out, 1.0)) * 255

        # 将图像像素转换为整数类型
        img_out_normalized = img_out_normalized.astype(cp.uint8)

        return img_out_normalized

    # 将输入数据转移到 GPU 上
    img_gpu = cp.asarray(rgb_img)
    # 调用处理函数
    result_gpu = process_channel(img_gpu)
    # 将结果数据从 GPU 转移回 CPU
    hist = cp.asnumpy(result_gpu)

    return hist


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


sigma = 0.033  # 标准差值，你可能需要根据应用场景进行调整
nbin = 64  # 直方图的桶数
input_dir = r"E:\桌面\1\1\val\images2"  # 输入图片文件夹路径
output_dir = r"E:\桌面\1\1\val\images3"  # 处理后直方图矩阵保存路径

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 遍历文件夹中的所有图片
for filename in os.listdir(input_dir):
    file_path = os.path.join(output_dir, filename)
    if os.path.exists(file_path):
        continue
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 图片完整路径
        img_path = os.path.join(input_dir, filename)
        print(img_path)
        # 读取图片
        rgb_img = cv_imread(img_path)
        # 转换为灰度图
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        # 图像压缩
        fx = 1.0
        fy = 1.0
        # 记录原始图片的尺寸
        original_height, original_width = gray_img.shape[:2]
        while True:
            try:
                # 图像压缩
                compressed_image = cv2.resize(gray_img, (0, 0), fx=fx, fy=fy)  # 将图像尺寸缩小
                # 调用LSH函数
                hist_mtx = LSH(compressed_image, sigma, nbin)

                break
            except Exception as e:
                if (fx <= 0.1):
                    fx *= 0.9
                    fy *= 0.9
                else:
                    fx -= 0.1
                    fy -= 0.1
                # 打印错误信息
                print(f"num {fx}: {fy}")
        # 图像放缩为原始尺寸
        resized_image = cv2.resize(hist_mtx, (original_width, original_height))  # 恢复原始尺寸

        # 转换直方图数组为图像
        img_final = Image.fromarray(resized_image, 'L')
        # 转换回RGB图像
        # 将灰度图转换为RGB图
        img_rgb = img_final.convert('RGB')
        # 指定保存文件的路径和文件名
        output_filename = os.path.join(output_dir, filename)

        # 保存图像到文件
        img_rgb.save(output_filename)
#              # 调用LSH函数
#             hist_mtx = LSH2(rgb_img, sigma, nbin)

#         # 转换直方图数组为图像
#         img_final = Image.fromarray(hist_mtx, 'RGB')

#         # 指定保存文件的路径和文件名
#         output_filename = os.path.join(output_dir, filename)
#         # 保存图像到文件
#         img_final.save(output_filename)
print("处理完成，所有直方图矩阵已保存至指定位置。")
