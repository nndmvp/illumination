import numpy as np
import cv2
import os
import torch
from PIL import Image
import cupy as cp
import traceback


# 相关系数
def count_diff2(q_curr, q_prev):
    sqrt_curr = (q_curr[:, 1] + q_curr[:, 2] + q_curr[:, 3]) / 3
    sqrt_prev = (q_prev[:, 1] + q_prev[:, 2] + q_prev[:, 3]) / 3
    tem1 = ((q_curr[:, 1] - sqrt_curr) * (q_prev[:, 1] - sqrt_prev)
            + (q_curr[:, 2] - sqrt_curr) * (q_prev[:, 2] - sqrt_prev)
            + (q_curr[:, 3] - sqrt_curr) * (q_prev[:, 3] - sqrt_prev))
    tem2 = cp.sqrt(((q_curr[:, 1] - sqrt_curr) ** 2 + (q_curr[:, 2] - sqrt_curr) ** 2 + (q_curr[:, 3] - sqrt_curr) ** 2)
                   * ((q_prev[:, 1] - sqrt_prev) ** 2 + (q_prev[:, 2] - sqrt_prev) ** 2 + (
                q_prev[:, 3] - sqrt_prev) ** 2))
    alpha_vector = tem1 / tem2
    alpha_vector = np.where(alpha_vector > 1, 1, alpha_vector)
    alpha_vector = np.where(alpha_vector < 0, 0, alpha_vector)
    return alpha_vector


def count_diff(q_curr, q_prev):
    tem1 = (q_curr[:, 1] - q_prev[:, 1]) ** 2 + (q_curr[:, 2] - q_prev[:, 2]) ** 2 + (
            q_curr[:, 3] - q_prev[:, 3]) ** 2 + 1
    tem2 = q_curr[:, 1] ** 2 + q_curr[:, 2] ** 2 + q_curr[:, 3] ** 2 + q_prev[:, 1] ** 2 + q_prev[:, 2] ** 2 + q_prev[
                                                                                                               :,
                                                                                                               3] ** 2
    alpha_vector = 1 - cp.exp(tem1 - tem2)
    alpha_vector = np.where(alpha_vector > 0.7, 0.7, alpha_vector)
    alpha_vector = np.where(alpha_vector < 0.3, 0.3, alpha_vector)
    return alpha_vector


def LSH_quaterion(rgb_img, sigma, nbin):
    # 定义颜色范围
    color_max = 255
    color_range = np.arange(0, color_max + 1, color_max / nbin)

    # 计算 alpha 值
    alpha_x = np.exp(-np.sqrt(2) / (sigma * rgb_img.shape[0]))
    alpha_y = np.exp(-np.sqrt(2) / (sigma * rgb_img.shape[1]))

    # 转为四元数
    quaternion_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=np.float64)
    quaternion_img[:, :, 1:] = rgb_img  # 将 RGB 值除以 255，转换为范围在 [0, 1] 的浮点数
    # 将输入数据转移到 GPU 上
    quaternion_img = cp.asarray(quaternion_img)

    q_mtx = cp.zeros((quaternion_img.shape[0], quaternion_img.shape[1], nbin, quaternion_img.shape[2]))

    for i in range(nbin):
        tmp_img = cp.array(quaternion_img[:, :, 1:])
        mask_l = tmp_img >= color_range[i]  # 获取大于等于下界的像素
        mask_u = tmp_img < color_range[i + 1]  # 获取小于上界的像素
        mask = cp.logical_and(mask_l, mask_u)  # 找到交集像素
        tmp_img[:] = 0
        tmp_img[mask] = 1
        q_mtx[:, :, i, 1:] = tmp_img

    # 初始化 hist_mtx 和 f_mtx
    hist_mtx = q_mtx.copy()
    f_mtx = cp.ones_like(q_mtx)

    norm = cp.sqrt(
        quaternion_img[:, :, 0] ** 2 +
        quaternion_img[:, :, 1] ** 2 +
        quaternion_img[:, :, 2] ** 2 +
        quaternion_img[:, :, 3] ** 2
    )
    norm_expanded = cp.expand_dims(norm, axis=2)
    # 归一化四元数
    quaternion_img_normalized = quaternion_img / norm_expanded
    # x 维度
    # 计算左部分
    hist_mtx_l = hist_mtx.copy()
    f_mtx_l = f_mtx.copy()
    for i in range(1, hist_mtx.shape[1]):
        # 提取归一化后的四元数的切片
        q_curr = quaternion_img_normalized[:, i, :]
        q_prev = quaternion_img_normalized[:, i - 1, :]

        alpha_q = count_diff(q_curr, q_prev)
        # # 计算乘积项
        # term1 = q_curr[:, 0] * q_prev[:, 0] + q_curr[:, 1] * q_prev[:, 1]
        # term2 = q_curr[:, 2] * q_prev[:, 2] + q_curr[:, 3] * q_prev[:, 3]
        #
        # # 求和得到最终结果
        # alpha_q = term1 + term2
        alpha_q = alpha_q[:, cp.newaxis, cp.newaxis]
        hist_mtx_l[:, i, :] += alpha_x * alpha_q * hist_mtx_l[:, i - 1, :]
        f_mtx_l[:, i, :] += alpha_x * alpha_q * f_mtx_l[:, i - 1, :]
    # 计算右部分
    hist_mtx_r = hist_mtx.copy()
    f_mtx_r = f_mtx.copy()
    for i in range(hist_mtx.shape[1] - 2, -1, -1):
        # 提取归一化后的四元数的切片
        q_curr = quaternion_img_normalized[:, i, :]
        q_prev = quaternion_img_normalized[:, i + 1, :]

        alpha_q = count_diff(q_curr, q_prev)
        alpha_q = alpha_q[:, cp.newaxis, cp.newaxis]
        hist_mtx_r[:, i, :] += alpha_x * alpha_q * hist_mtx_r[:, i + 1, :]
        f_mtx_r[:, i, :] += alpha_x * alpha_q * f_mtx_r[:, i + 1, :]
    # 合并右部分和左部分
    hist_mtx = hist_mtx_r + hist_mtx_l - q_mtx
    f_mtx = f_mtx_r + f_mtx_l - 1

    # y 维度
    # 计算左部分
    hist_mtx_l = hist_mtx.copy()
    f_mtx_l = f_mtx.copy()
    for i in range(1, hist_mtx.shape[0]):
        # 提取归一化后的四元数的切片
        q_curr = quaternion_img_normalized[i, :, :]
        q_prev = quaternion_img_normalized[i - 1, :, :]

        alpha_q = count_diff(q_curr, q_prev)
        alpha_q = alpha_q[:, cp.newaxis, cp.newaxis]
        hist_mtx_l[i, :, :] += alpha_y * alpha_q * hist_mtx_l[i - 1, :, :]
        f_mtx_l[i, :, :] += alpha_y * alpha_q * f_mtx_l[i - 1, :, :]
    # 计算右部分
    hist_mtx_r = hist_mtx.copy()
    f_mtx_r = f_mtx.copy()
    for i in range(hist_mtx.shape[0] - 2, -1, -1):
        # 提取归一化后的四元数的切片
        q_curr = quaternion_img_normalized[i, :, :]
        q_prev = quaternion_img_normalized[i + 1, :, :]

        alpha_q = count_diff(q_curr, q_prev)
        alpha_q = alpha_q[:, cp.newaxis, cp.newaxis]
        hist_mtx_r[i, :, :] += alpha_y * alpha_q * hist_mtx_r[i + 1, :, :]
        f_mtx_r[i, :, :] += alpha_y * alpha_q * f_mtx_r[i + 1, :, :]
    # 合并右部分和左部分
    hist_mtx = hist_mtx_r + hist_mtx_l - q_mtx
    f_mtx = f_mtx_r + f_mtx_l - 1
    # 使用归一化因子对 H 进行归一化
    hist_mtx /= f_mtx
    k = 0.033
    step = color_max / nbin
    # 计算 rp1
    rp1 = k * quaternion_img

    # 获取当前的 bin 值
    bp = quaternion_img // step

    # 计算 b 的范围
    bins = cp.arange(nbin)

    # 计算直方图权重
    # 计算最大值
    max_value = cp.maximum(k, rp1)
    # 扩展 max_value 的维度，使其与 numerator 的形状相匹配
    max_value_expanded = cp.expand_dims(max_value, axis=2)
    bins_m = cp.expand_dims(bins, axis=(0, 1, 3))
    # 计算分子
    numerator = cp.square(bins_m - cp.expand_dims(bp, axis=2))
    # 计算权重
    hist_weights = cp.exp(-numerator / (2 * cp.square(max_value_expanded)))

    # 计算 Ip
    Ip = cp.sum(hist_mtx * hist_weights, axis=2)

    # 将 Ip 赋值给输出图像
    img_out = Ip
    # 将图像像素归一化到 [0, 255] 范围内
    img_out_normalized = ((img_out - cp.min(img_out)) / (cp.max(img_out) - cp.min(img_out))) * 255

    # 将图像像素转换为整数类型
    img_out_normalized = img_out_normalized.astype(cp.uint8)
    hist = cp.asnumpy(img_out_normalized[:, :, 1:])
    return hist


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


sigma = 0.033  # 标准差值，你可能需要根据应用场景进行调整
nbin = 64  # 直方图的桶数
input_dir = r"F:\电网数据集\绝缘子歪斜1026\images"  # 输入图片文件夹路径
output_dir = r"E:\桌面\1\1\train\images5"  # 处理后直方图矩阵保存路径
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
        if (rgb_img.shape[2] != 3):
            continue
        # # 图像压缩
        fx = 1
        fy = 1
        while True:
            try:

                # 记录原始图片的尺寸
                original_height, original_width = rgb_img.shape[:2]

                # 图像压缩
                compressed_image = cv2.resize(rgb_img, None, fx=fx, fy=fy)  # 将图像尺寸缩小一半
                # 调用LSH函数
                hist_mtx = LSH_quaterion(compressed_image, sigma, nbin)

                # 图像放缩为原始尺寸
                resized_image = cv2.resize(hist_mtx, (original_width, original_height))  # 恢复原始尺寸

                # 转换直方图数组为图像
                img_final = Image.fromarray(resized_image, 'RGB')
                # 指定保存文件的路径和文件名
                output_filename = os.path.join(output_dir, filename)

                # 保存图像到文件
                img_final.save(output_filename)
                break
            except Exception as e:
                print(f"Exception occurred: {e}")
                if (fx <= 0.21):
                    fx *= 0.9
                    fy *= 0.9
                else:
                    fx -= 0.1
                    fy -= 0.1
                # 打印错误信息
                print(f"num {fx}: {fy}")
#              # 调用LSH函数
#             hist_mtx = LSH2(rgb_img, sigma, nbin)

#         # 转换直方图数组为图像
#         img_final = Image.fromarray(hist_mtx, 'RGB')

#         # 指定保存文件的路径和文件名
#         output_filename = os.path.join(output_dir, filename)

#         # 保存图像到文件
#         img_final.save(output_filename)
print("处理完成，所有直方图矩阵已保存至指定位置。")
