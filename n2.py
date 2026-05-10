import math
import time
import numpy as np
import cv2
import os
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import Sequential, AdaptiveAvgPool2d
from pytorch_wavelets import DWTForward, DWTInverse
from pytorch_msssim import MS_SSIM
from torch.utils.data import Dataset
import models as Model
import argparse
import core.logger as Logger
from torch.utils.data import DataLoader
import os
from models.fs_modules.fs_head import Fusion_Head
import random
from torch.nn import Parameter
from torchvision.transforms.functional import to_tensor
from data.util import *
from torchvision.transforms import functional as F2


class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.hr_transform = train_hr_transform(160)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.1)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img_path = os.path.join(self.input_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        # 读取输入图像和 ground truth 图像
        rgb_img = Image.open(img_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        crop_size = self.hr_transform(rgb_img)
        rgb_img, gt_img = F2.crop(rgb_img, crop_size[0], crop_size[1],
                                  crop_size[2],
                                  crop_size[3]), \
            F2.crop(gt_img, crop_size[0], crop_size[1], crop_size[2], crop_size[3])

        if random.random() > 0.5:
            rgb_img = self.hflip(rgb_img)
            gt_img = self.hflip(gt_img)

        if random.random() > 0.5:
            rgb_img = self.vflip(rgb_img)
            gt_img = self.vflip(gt_img)

        # 转换为 NumPy 数组
        rgb_img = np.array(rgb_img)
        gt_img = np.array(gt_img)

        # 如果有额外的通道（如 Alpha 通道），只保留前 3 个通道
        if rgb_img.shape[2] != 3:
            rgb_img = rgb_img[:, :, :3]
        if gt_img.shape[2] != 3:
            gt_img = gt_img[:, :, :3]

        # 应用变换（如果有的话）
        if self.transform:
            rgb_img = self.transform(rgb_img)
            gt_img = self.transform(gt_img)

        return rgb_img, gt_img, filename


class ImageDatasetTest(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img_path = os.path.join(self.input_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        # 读取输入图像和 ground truth 图像
        rgb_img = Image.open(img_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        # 转换为 NumPy 数组
        rgb_img = np.array(rgb_img)
        gt_img = np.array(gt_img)

        # 如果有额外的通道（如 Alpha 通道），只保留前 3 个通道
        if rgb_img.shape[2] != 3:
            rgb_img = rgb_img[:, :, :3]
        if gt_img.shape[2] != 3:
            gt_img = gt_img[:, :, :3]

        # 应用变换（如果有的话）
        if self.transform:
            rgb_img = self.transform(rgb_img)
            gt_img = self.transform(gt_img)

        return rgb_img, gt_img, filename


class QuaternionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(QuaternionConv2d, self).__init__()
        self.in_channels = in_channels // 4  # 四元数的四个分量 (r, i, j, k)
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.dilation = dilation  # 新增的膨胀系数参数

        # 初始化四元数权重 (4, out_channels, in_channels, kH, kW)
        self.weight = nn.Parameter(
            torch.Tensor(4, self.out_channels, self.in_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # 拆分输入为四元数组件 (batch, in_channels, H, W) -> 4 x (batch, in_channels//4, H, W)
        x_r, x_i, x_j, x_k = torch.chunk(x, 4, dim=1)

        # 拆分权重 (4, out_channels, in_channels, kH, kW)
        w_r, w_i, w_j, w_k = self.weight

        # 构造拼接后的权重 (4*out_channels, in_channels, kH, kW)
        cat_kernels_r = torch.cat([w_r, -w_i, -w_j, -w_k], dim=1)  # 实部权重
        cat_kernels_i = torch.cat([w_i, w_r, -w_k, w_j], dim=1)  # i分量权重
        cat_kernels_j = torch.cat([w_j, w_k, w_r, -w_i], dim=1)  # j分量权重
        cat_kernels_k = torch.cat([w_k, -w_j, w_i, w_r], dim=1)  # k分量权重

        # 最终拼接 (4*out_channels, 4*in_channels, kH, kW)
        cat_kernels = torch.cat([cat_kernels_r, cat_kernels_i, cat_kernels_j, cat_kernels_k], dim=0)

        # 拼接输入 (batch, 4*in_channels, H, W)
        cat_input = torch.cat([x_r, x_i, x_j, x_k], dim=1)

        # 使用dilation参数的卷积计算
        out = F.conv2d(
            cat_input,
            cat_kernels,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation  # 添加膨胀系数
        )

        return out


class QuaternionBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        四元数实例归一化层

        参数:
            num_features: 输入通道数（必须是4的倍数）
            eps: 数值稳定项
            affine: 是否使用可学习的仿射参数
        """
        super(QuaternionBatchNorm2d, self).__init__()
        assert num_features % 4 == 0, "num_features must be divisible by 4"
        self.num_features = num_features
        self.quat_features = num_features // 4  # 四元数数量
        self.eps = eps
        self.affine = affine

        if affine:
            # 可学习的仿射参数 (gamma, beta)
            # 每个四元数的四个分量共享相同的缩放和偏移参数
            self.gamma = nn.Parameter(torch.ones(1, self.quat_features, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.quat_features, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        # 输入形状: (batch, 4*quat_features, H, W)
        # 重塑为: (batch, quat_features, 4, H, W)
        x_reshaped = x.view(x.size(0), self.quat_features, 4, x.size(2), x.size(3))

        # 计算模长 (保持维度用于广播)
        modulus = torch.sqrt(torch.sum(x_reshaped ** 2, dim=2, keepdim=True) + self.eps)

        # 归一化: 每个四元数除以其模长
        normalized = x_reshaped / modulus

        # 应用仿射变换 (如果启用)
        if self.affine:
            normalized = self.gamma.unsqueeze(2) * normalized + self.beta.unsqueeze(2)

        # 恢复原始形状
        return normalized.view(x.size())


class QuaternionLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.1, inplace=False, enhance_low_contrast=False):
        super(QuaternionLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.enhance_low_contrast = enhance_low_contrast  # 新增：是否增强低对比度区域

    def forward(self, x):
        if x.size(1) % 4 != 0:
            raise ValueError("Input channels must be divisible by 4 for quaternion LeakyReLU.")

        chunks = torch.chunk(x, 4, dim=1)
        activated_chunks = []

        for chunk in chunks:
            activated = F.leaky_relu(chunk, negative_slope=self.negative_slope, inplace=self.inplace)

            # 增强低对比度区域的响应
            if self.enhance_low_contrast:
                # 计算局部方差作为对比度度量
                local_var = F.avg_pool2d(chunk ** 2, kernel_size=3, padding=1, stride=1) - \
                            F.avg_pool2d(chunk, kernel_size=3, padding=1, stride=1) ** 2

                # 低对比度区域的掩码（方差小于平均值）
                low_contrast_mask = (local_var < torch.mean(local_var, dim=[1, 2, 3], keepdim=True)).float()

                # 增强低对比度区域的响应
                contrast_enhancement = 2.0 * (torch.sigmoid(activated) - 1.0)
                activated = activated + low_contrast_mask * contrast_enhancement

            activated_chunks.append(activated)

        return torch.cat(activated_chunks, dim=1)


class QuaternionECACond(nn.Module):
    def __init__(self, channel, reduction=16, cond_dim=0, gamma=2, b=1):
        """
        带条件输入的高效四元数通道注意力 (Quaternion ECA with Condition)

        :param channel: 输入通道数（四元数表示前的通道数）
        :param reduction: 通道压缩比例
        :param cond_dim: 条件输入维度（四元数表示前的通道数）
        :param gamma, b: 自适应卷积核大小参数
        """
        super().__init__()
        self.channel = channel
        self.cond_dim = cond_dim

        # 自适应计算卷积核大小
        t = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1

        # 自适应全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 条件融合层
        if cond_dim > 0:
            # 使用四元数卷积处理条件输入
            self.cond_fusion = nn.Sequential(
                QuaternionConv2d(cond_dim, channel // reduction, kernel_size=1),
                QuaternionLeakyReLU(0.2),
                QuaternionBatchNorm2d(channel // reduction),
                QuaternionConv2d(channel // reduction, channel, kernel_size=1),
                QuaternionLeakyReLU(0.2)
            )

        # 高效通道注意力模块
        # 注意：输入通道数为1，因为我们处理的是标量权重
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(1)
        )
        # 激活函数
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = QuaternionLeakyReLU(0.2)

    def forward(self, x, cond=None):
        """
        :param x: 输入特征 [B, 4*C, H, W]（四元数表示）
        :param cond: 条件输入 [B, 4*C_cond, H, W] 或 None
        :return: 注意力加权后的特征
        """
        b, c, h, w = x.size()
        c_quat = c // 4  # 四元数通道数

        # 1. 计算四元数模长作为注意力基础
        r, i, j, k = torch.chunk(x, 4, dim=1)
        magnitude = torch.sqrt(r ** 2 + i ** 2 + j ** 2 + k ** 2 + 1e-8)

        # 2. 全局平均池化
        y = self.avg_pool(magnitude)  # [B, C_quat, 1, 1]

        # 3. 条件信息融合
        if cond is not None and hasattr(self, 'cond_fusion'):
            # 对条件输入进行全局平均池化
            cond_pooled = self.avg_pool(cond)  # [B, 4*C_cond, 1, 1]

            # 使用四元数卷积融合条件信息
            cond_weights = self.cond_fusion(cond_pooled)  # [B, 4*C, 1, 1]

            # 提取条件权重中的实部作为注意力调制因子
            cond_r, _, _, _ = torch.chunk(cond_weights, 4, dim=1)

            # 与主注意力融合（乘法融合）
            y = y * cond_r

        # 4. 调整维度用于1D卷积
        # 现在y的形状是[B, C_quat, 1, 1]
        # 我们需要将其转换为[B, 1, C_quat]用于1D卷积
        y = y.view(b, c_quat)  # [B, C_quat]
        y = y.unsqueeze(1)  # [B, 1, C_quat]

        # 5. 应用1D卷积进行跨通道交互
        y = self.conv(y)  # [B, 1, C_quat]

        # 6. 调整回原始维度
        y = y.squeeze(1)  # [B, C_quat]
        y = y.view(b, c_quat, 1, 1)  # [B, C_quat, 1, 1]

        # 7. 应用激活函数和sigmoid
        y = self.activation(y)
        y = self.sigmoid(y)

        # 8. 扩展到四元数的四个分量
        weights = y.repeat(1, 4, 1, 1)  # [B, 4*C_quat, 1, 1]

        # 9. 应用注意力
        return x * weights


class DepthwiseSeparableQuaternionConv(nn.Module):
    """深度可分离四元数卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 添加组归一化
        self.depthwise = nn.Sequential(
            QuaternionConv2d(in_channels, in_channels, kernel_size, stride, padding),
            QuaternionBatchNorm2d(in_channels),
            QuaternionLeakyReLU(0.2)
        )
        self.pointwise = nn.Sequential(
            QuaternionConv2d(in_channels, out_channels, 1),
            QuaternionBatchNorm2d(out_channels)
        )

    def forward(self, x):
        o = self.pointwise(self.depthwise(x))
        return o


class InvertedResidualQuaternionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        expanded_channels = in_channels * expansion

        # 改进的扩展层
        self.expand = nn.Sequential(
            QuaternionConv2d(in_channels, expanded_channels, 1),
            QuaternionBatchNorm2d(expanded_channels),
            QuaternionLeakyReLU(0.2)
        ) if expansion > 1 else nn.Identity()

        # 改进的深度可分离卷积
        self.depthwise = nn.Sequential(
            DepthwiseSeparableQuaternionConv(expanded_channels, expanded_channels, 3),
            QuaternionBatchNorm2d(expanded_channels),
            QuaternionECACond(expanded_channels)
        )

        # 改进的投影层
        self.project = nn.Sequential(
            QuaternionConv2d(expanded_channels, out_channels, 1),
            QuaternionBatchNorm2d(out_channels)
        )

        # 改进的残差连接
        self.residual = nn.Identity() if in_channels == out_channels else None
        self.skip_gain = nn.Parameter(torch.zeros(1))  # 可学习的残差强度

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        if self.residual is not None:
            return identity + self.skip_gain * out
        return out


class LowRankGlobalAttention(nn.Module):
    """改进版：引入距离衰减的全局注意力机制（支持负值）"""

    def __init__(self, in_channels, reduction_ratio=4, num_landmarks=32, base_scale=10.0):
        super(LowRankGlobalAttention, self).__init__()
        # 保持原始输入通道数（完整的四元数通道）
        self.in_channels = in_channels
        self.reduced_channels = max(4, self.in_channels // reduction_ratio)
        self.num_landmarks = num_landmarks

        # 关键点生成网络 - 使用完整四元数输入
        # 改进1：增强的关键点生成
        self.landmark_generator = nn.Sequential(
            InvertedResidualQuaternionBlock(in_channels, num_landmarks * 4),
            QuaternionConv2d(num_landmarks * 4, num_landmarks, 1),
            QuaternionLeakyReLU(0.2)  # 限制关键点范围
        )

        # 局部上下文提取 - 使用完整四元数输入
        self.local_context = nn.Sequential(
            QuaternionConv2d(self.in_channels, self.reduced_channels, kernel_size=3, padding=1, stride=1),
            QuaternionBatchNorm2d(self.reduced_channels),
            QuaternionLeakyReLU(0.2)
        )

        # 全局信息投影 - 输出与输入相同的四元数通道
        self.global_proj = nn.Sequential(
            DepthwiseSeparableQuaternionConv(self.reduced_channels, self.in_channels, kernel_size=1, padding=0,
                                             stride=1),
            QuaternionECACond(self.in_channels, self.in_channels, cond_dim=0),
            QuaternionLeakyReLU(0.2)
        )
        # 自适应融合参数
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        # 距离衰减参数
        self.distance_scale = nn.Parameter(torch.tensor(base_scale))  # 可学习的距离缩放因子

    def forward(self, x):
        batch_size, _, H, W = x.shape

        # 创建归一化坐标网格
        with torch.no_grad():
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(0, 1, H, device=x.device),
                torch.linspace(0, 1, W, device=x.device),
                indexing='ij'
            )
            coord_grid = torch.stack((x_coords, y_coords), dim=-1)  # [H, W, 2]
            coord_grid = coord_grid.view(-1, 2)  # [H*W, 2]

        # 直接在整个四元数特征上操作（不再拆分）
        # 1. 生成关键点（注意力锚点）
        logits = self.landmark_generator(x).view(batch_size, self.num_landmarks, -1)  # [B, num_landmarks, H*W]

        # 2. 双极性归一化
        positive_logits = torch.relu(logits)
        negative_logits = -torch.relu(-logits)
        positive_weights = F.softmax(positive_logits, dim=-1)
        negative_weights = F.softmax(negative_logits, dim=-1)
        landmarks = positive_weights - negative_weights  # [B, num_landmarks, H*W]

        # 3. 计算关键点期望坐标
        landmark_coords = torch.matmul(landmarks, coord_grid)  # [B, num_landmarks, 2]

        # 4. 计算距离权重矩阵
        positions = coord_grid.view(1, 1, H * W, 2)  # [1, 1, H*W, 2]
        centers = landmark_coords.view(batch_size, self.num_landmarks, 1, 2)  # [B, num_landmarks, 1, 2]
        dist = torch.norm(positions - centers, dim=-1)  # [B, num_landmarks, H*W]

        # 5. 应用距离衰减
        distance_weights = torch.exp(-dist * torch.exp(self.distance_scale))

        # 6. 将距离权重融入landmarks
        weighted_landmarks = landmarks * distance_weights
        # weighted_landmarks = F.softmax(weighted_landmarks, dim=-1)  # 保持归一化

        # 7. 提取局部上下文特征
        local_feat = self.local_context(x)  # [B, C_red, H, W]
        local_feat_flat = local_feat.view(batch_size, self.reduced_channels, -1)  # [B, C_red, H*W]

        # 8. 通过关键点聚合全局信息
        global_info = torch.bmm(local_feat_flat, weighted_landmarks.transpose(1, 2))

        # 9. 扩散全局信息到所有位置
        global_info = torch.bmm(global_info, weighted_landmarks)
        global_info = global_info.view(batch_size, self.reduced_channels, H, W)

        # 10. 投影全局信息
        global_info = self.global_proj(global_info)  # [B, C, H, W]

        # 11. 自适应融合
        out = self.alpha * global_info + self.beta * x

        return out


class EnhancedConditionalProcessing(nn.Module):
    """增强型条件处理模块，包含多分支门控机制"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main_branch = nn.Sequential(
            QuaternionConv2d(in_channels, out_channels * 2, kernel_size=3, padding=1),
            QuaternionBatchNorm2d(out_channels * 2),
            QuaternionLeakyReLU(0.2)
        )

        # 门控分支
        self.gate_branch = nn.Sequential(
            QuaternionConv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 残差路径
        self.res_path = nn.Sequential(
            QuaternionConv2d(in_channels, out_channels, kernel_size=1),
            QuaternionBatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        main = self.main_branch(x)
        gate = self.gate_branch(x)

        # 将主分支分为内容和门控两部分
        content, modulation = torch.chunk(main, 2, dim=1)
        modulated = content * (1 + modulation)  # 增强特征表达

        return self.res_path(x) + gate * modulated


class EnhancedChannelAttention(nn.Module):
    """增强型通道注意力，包含通道交互与门控机制"""

    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        inter_channels = max(4, channels // reduction_ratio)

        self.channel_interaction = nn.Sequential(
            QuaternionConv2d(channels * 2, inter_channels, kernel_size=1),
            QuaternionLeakyReLU(0.2),
            QuaternionConv2d(inter_channels, channels, kernel_size=1),
            nn.Tanh()
        )

        self.adaptive_gate = nn.Sequential(
            QuaternionConv2d(channels, channels, kernel_size=1),
            QuaternionLeakyReLU(0.2)
        )

    def forward(self, x):
        # 通道间交互
        identity = x
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        interacted = self.channel_interaction(pooled)

        # 生成自适应门控
        gate = self.adaptive_gate(interacted)

        # 门控残差连接
        return identity + identity * gate


class HeadLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadLeakyRelu2d, self).__init__()
        self.conv = QuaternionConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                     stride=stride)
        self.re = QuaternionLeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.re(self.conv(x))


def sum_quaternion_components(quat_tensor):
    """
    对四元数张量的各分量（w, i, j, k）分别求和。

    参数:
        quat_tensor (torch.Tensor): 形状为 [1, n*4, W, H] 的四元数张量，
            通道顺序为 [w1, i1, j1, k1, w2, i2, j2, k2, ..., wn, in, jn, kn].

    返回:
        torch.Tensor: 形状为 [1, 4, W, H] 的张量，表示各分量的总和。
    """
    # 检查输入形状
    B, C, W, H = quat_tensor.shape
    n = C // 4  # 四元数数量

    # 提取所有 w/i/j/k 分量 (通过切片步长实现)
    w = quat_tensor[:, 0::4, :, :]  # 形状 [1, n, W, H]
    i = quat_tensor[:, 1::4, :, :]  # 形状 [1, n, W, H]
    j = quat_tensor[:, 2::4, :, :]  # 形状 [1, n, W, H]
    k = quat_tensor[:, 3::4, :, :]  # 形状 [1, n, W, H]

    # 对各分量求和
    w_sum = torch.sum(w, dim=1, keepdim=True)  # [1, 1, W, H]
    i_sum = torch.sum(i, dim=1, keepdim=True)  # [1, 1, W, H]
    j_sum = torch.sum(j, dim=1, keepdim=True)  # [1, 1, W, H]
    k_sum = torch.sum(k, dim=1, keepdim=True)  # [1, 1, W, H]

    # 拼接结果 [1, 4, W, H]
    result = torch.cat([w_sum, i_sum, j_sum, k_sum], dim=1)

    return result


class QuaternionDropout(nn.Module):
    """四元数感知的Dropout层"""

    def __init__(self, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        n, c, h, w = x.shape
        x_grouped = x.view(n, c // 4, 4, h, w)
        mask = torch.ones(n, c // 4, 1, 1, device=x.device)
        mask = self.dropout(mask)
        return (x_grouped * mask.unsqueeze(2)).view(n, c, h, w)


class Predictor(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super().__init__()
        # Modified first convolution to accept concatenated inputs (img + r_h)
        self.conv1 = QuaternionConv2d(in_channels=4,  # 4 (img) + 4 (r_h)
                                      out_channels=8,
                                      kernel_size=3,
                                      padding=1)
        self.initial_activation = QuaternionLeakyReLU(0.2)
        self.cond_processing = EnhancedConditionalProcessing(4, 32)

        self.conv2 = QuaternionConv2d(8, 32, kernel_size=3, padding=1)
        self.bn2 = QuaternionBatchNorm2d(32)
        self.act2 = QuaternionLeakyReLU(0.2)  # 保持负斜率
        # 注意力机制
        self.spatial_att = LowRankGlobalAttention(32)

        # 改进的通道注意力（支持负值）
        self.channel_att = EnhancedChannelAttention(32, reduction_ratio=4)

        # 双极性注意力融合
        self.att_fusion = nn.Sequential(
            QuaternionConv2d(32 * 2, 32, kernel_size=1),
            QuaternionLeakyReLU(0.2)
        )

        self.conv3 = QuaternionConv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = QuaternionBatchNorm2d(32)
        self.act3 = QuaternionLeakyReLU(0.2)  # 保持负斜率
        self.dropout = QuaternionDropout(0.2)

    def forward(self, fds, img):
        # Process through initial convolution
        init_conv = self.conv1(img)
        init_conv = self.initial_activation(init_conv)

        # Pass through residual block with original img as condition
        cond_feat = self.cond_processing(img)
        # 双极性门控（-1到1范围）
        gate = torch.tanh(cond_feat)

        # 主路径处理
        out = self.conv2(init_conv)
        out = self.bn2(out)
        out = self.act2(out * gate)
        identity = out
        # 双注意力
        spatial_out = self.spatial_att(out)

        # 改进的通道注意力（支持负值）
        channel_att = self.channel_att(out)
        channel_out = out + channel_att  # 残差连接保持负值

        # 双极性融合
        att_features = torch.cat([spatial_out, channel_out], dim=1)
        att_weights = self.att_fusion(att_features)
        att_weights = torch.sigmoid(att_weights)
        fused_att = att_weights * spatial_out + (1 - att_weights) * channel_out

        # 残差连接
        out = self.bn3(self.conv3(fused_att))
        out = self.act3(identity + out)
        features = self.dropout(out)  # 最终激活保留负值

        # output = sum_quaternion_components(features)
        intermediate = sum_quaternion_components(features)

        return features, intermediate


# 读取图像的函数
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        # 定义 Sobel 算子卷积核
        kernelx = torch.FloatTensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, img1, img2):
        """
        计算两个图像之间的梯度损失。

        参数:
            img1: 第一个输入图像 (batch_size, channels, height, width)
            img2: 第二个输入图像 (batch_size, channels, height, width)

        返回:
            loss: 两个图像梯度之间的平均绝对误差
        """
        batch_size = img1.size()[0]
        grad_loss = 0.0

        for i in range(img1.shape[1]):  # 对每个通道分别计算梯度
            # 计算img1的水平和垂直梯度
            gx_img1 = F.conv2d(img1[:, i:i + 1], self.weightx, padding=1)
            gy_img1 = F.conv2d(img1[:, i:i + 1], self.weighty, padding=1)

            # 计算img2的水平和垂直梯度
            gx_img2 = F.conv2d(img2[:, i:i + 1], self.weightx, padding=1)
            gy_img2 = F.conv2d(img2[:, i:i + 1], self.weighty, padding=1)

            # 计算梯度幅值
            grad_img1 = torch.sqrt(torch.pow(gx_img1, 2) + torch.pow(gy_img1, 2) + 1e-6)
            grad_img2 = torch.sqrt(torch.pow(gx_img2, 2) + torch.pow(gy_img2, 2) + 1e-6)

            # 计算梯度之间的差异，并求平均
            diff = torch.abs(grad_img1 - grad_img2)
            grad_loss += torch.mean(diff)

        return grad_loss / img1.shape[1]  # 平均所有通道的损失


def angle(a, b, eps=1e-8):
    """
    计算两个向量之间的夹角（弧度制）
    Args:
        a: 第一个向量，形状为 (N, H, W)
        b: 第二个向量，形状为 (N, H, W)
        eps: 小常数，用于数值稳定性
    Returns:
        theta: 夹角，形状为 (N)
    """
    vector = a * b
    up = torch.sum(vector, dim=(1, 2))  # (N)
    down = torch.norm(a, p=2, dim=(1, 2)) * torch.norm(b, p=2, dim=(1, 2))  # (N)
    cos_theta = up / (down + eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)  # 防止 acos 输入超出范围
    theta = torch.acos(cos_theta)  # 弧度制
    return theta


def color_loss(out_image, gt_image, eps=1e-8):
    """
    计算颜色一致性损失
    Args:
        out_image: 融合后的图像，形状为 (N, C, H, W)
        gt_image: 原始可见图像，形状为 (N, C, H, W)
        eps: 小常数，用于数值稳定性
    Returns:
        loss: 颜色一致性损失
    """
    N, C, H, W = out_image.size()
    loss = 0.0
    for i in range(C):
        loss += torch.mean(angle(out_image[:, i, :, :], gt_image[:, i, :, :], eps))
    loss /= C
    return loss


class TVLoss(nn.Module):
    """
    Total Variation Loss
    """

    def __init__(self, weight=1.0, mode='aniso'):
        """
        初始化 TV Loss

        参数:
            weight (float): TV Loss 的权重
            mode (str): 'aniso' 或 'iso'，选择各向异性或各向同性TV Loss
        """
        super(TVLoss, self).__init__()
        self.weight = weight
        self.mode = mode

    def forward(self, x):
        """
        计算 TV Loss

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, channels, height, width]

        返回:
            torch.Tensor: TV Loss 值
        """
        batch_size, channels, height, width = x.size()

        # 计算水平方向的差异
        h_diff = x[:, :, 1:, :] - x[:, :, :-1, :]

        # 计算垂直方向的差异
        w_diff = x[:, :, :, 1:] - x[:, :, :, :-1]

        if self.mode == 'aniso':
            # 各向异性 TV Loss
            loss = torch.abs(h_diff).sum() + torch.abs(w_diff).sum()
        elif self.mode == 'iso':
            # 各向同性 TV Loss
            loss = torch.sqrt(h_diff.pow(2) + w_diff.pow(2) + 1e-8).sum()
        else:
            raise ValueError("mode 必须是 'aniso' 或 'iso'")

        return self.weight * loss / (batch_size * channels * height * width)


tv_loss = TVLoss()


def to_quaternion(L):
    """
    将 LL 子带（PyTorch 张量）转换为四元数形式（实部为 0，虚部分别为 R、G、B）
    输入: LL, 形状为 [batch_size, 3, H, W]
    输出: quaternion_LL, 形状为 [batch_size, 4, H, W]
    """
    batch_size, _, h, w = L.shape
    # 创建实部为 0 的四元数张量
    quaternion_L = torch.zeros(batch_size, 4, h, w, device=L.device)
    # 虚部分别为 R、G、B
    quaternion_L[:, 1:, :, :] = L
    return quaternion_L


class VGGLoss(nn.Module):
    def __init__(self, layers=[5, 10, 19], device=None):  # 修改了默认的 layers 参数
        super(VGGLoss, self).__init__()

        # 加载预训练的 VGG19 模型，并移除分类器部分，只保留特征提取部分
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features  # 使用 VGG19

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        vgg = vgg.to(self.device)

        self.feature_layers = layers  # 可以根据需要调整这些层索引
        self.feature_extractors = nn.ModuleList()

        for layer_index in self.feature_layers:
            extractor = nn.Sequential(*list(vgg.children())[:layer_index])
            extractor.to(self.device)
            self.feature_extractors.append(extractor)

        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

        # 使用 L1 损失函数
        self.l1_loss = nn.L1Loss().to(self.device)

    def forward(self, input_image, target_image):
        # 确保输入和目标图像在同一设备上
        input_image = input_image.to(self.device)
        target_image = target_image.to(self.device)

        input_features = []
        target_features = []

        with torch.no_grad():  # 禁用梯度计算以提高效率
            for extractor in self.feature_extractors:
                input_features.append(extractor(input_image))

        for extractor in self.feature_extractors:
            target_features.append(extractor(target_image))

        loss = 0
        for input_feat, target_feat in zip(input_features, target_features):
            loss += self.l1_loss(input_feat, target_feat.detach())

        return loss


def ll_subband_ssim_loss(ll_output, ll_target):
    """
    ll_output: 模型输出的LL子带系数 (形状: [B, C, H, W])
    ll_target: 目标LL子带系数 (形状: [B, C, H, W])
    """
    ms_ssim_loss_fn = MS_SSIM(data_range=1.0, size_average=True, channel=ll_output.shape[1])
    return 1 - ms_ssim_loss_fn(ll_output, ll_target)


class SobelEdgeLoss(nn.Module):
    def __init__(self, alpha=0.7, theta=5.0, epsilon=1e-6):
        """
        高频子带边缘损失
        :param alpha: 强度损失权重 (0-1)
        :param theta: 损失缩放系数
        :param epsilon: 数值稳定项
        """
        super().__init__()
        self.alpha = alpha
        self.theta = theta
        self.epsilon = epsilon

        # 注册Sobel算子为缓冲张量
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)  # 归一化

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)

    def compute_edges(self, x):
        """
        计算边缘强度图和梯度方向
        :param x: 输入张量 [B,C,H,W]
        :return: (强度图, 梯度方向余弦)
        """
        b, c, h, w = x.size()

        # 各通道独立计算
        strength_maps = []
        direction_cos = []

        for ch in range(c):
            channel = x[:, ch:ch + 1]  # [B,1,H,W]

            # Sobel梯度
            gx = F.conv2d(channel, self.sobel_x, padding=1)
            gy = F.conv2d(channel, self.sobel_y, padding=1)

            # 边缘强度
            strength = torch.sqrt(gx ** 2 + gy ** 2 + self.epsilon)
            strength_maps.append(strength)

            # 梯度方向 (归一化余弦值)
            norm = torch.sqrt(gx ** 2 + gy ** 2 + self.epsilon)
            dir_x = gx / norm
            dir_y = gy / norm
            direction_cos.append(torch.stack([dir_x, dir_y], dim=2))  # [B,1,2,H,W]

        return torch.cat(strength_maps, dim=1), torch.cat(direction_cos, dim=1)

    def forward(self, pred, target):
        """
        :param pred: 预测的高频子带 [B,C,H,W]
        :param target: 目标高频子带 [B,C,H,W]
        :return: 复合边缘损失
        """
        # ==================== 边缘强度计算 ====================
        pred_strength, pred_dir = self.compute_edges(pred)
        target_strength, target_dir = self.compute_edges(target)

        # ==================== 强度相似性损失 ====================
        strength_loss = F.l1_loss(pred_strength,
                                  target_strength)

        # ==================== 方向一致性损失 ====================
        # 余弦相似度计算
        dot_product = (pred_dir * target_dir).sum(dim=2)  # [B,C,H,W]
        direction_loss = 1 - dot_product.mean()

        # ==================== 复合损失 ====================
        total_loss = (
                             self.alpha * strength_loss +
                             (1 - self.alpha) * direction_loss
                     ) * self.theta

        return total_loss


import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/fusion_train.json',
                    help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                    help='Run either train(training + validation) or testing', default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_eval', action='store_true')

# Parse configs
args = parser.parse_args()
opt = Logger.parse(args)
opt = Logger.dict_to_nonedict(opt)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model initialization
df_model_opt = opt['model_df']
diffusion_model_opt = opt['model']
fusion_head = Fusion_Head(
    feat_scales=df_model_opt['feat_scales'],
    out_channels=df_model_opt['out_channels'],
    inner_channel=diffusion_model_opt['unet']['inner_channel'],
    channel_multiplier=diffusion_model_opt['unet']['channel_multiplier'],
    img_size=df_model_opt['output_cm_size'],
    time_steps=df_model_opt["t"]
).to(device)

# Loading diffusion model
diffusion = Model.create_model(opt)
print(f"模型参数量统计:")
print(f"========================")
total_params = sum(p.numel() for p in fusion_head.parameters())
trainable_params = sum(p.numel() for p in fusion_head.parameters() if p.requires_grad)
print(f"Fusion_Head 总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
print(f"Fusion_Head 可训练参数: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
print(f"========================")
# Define models

# Loss functions and optimizer
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
vgg_loss = VGGLoss()
gradient_loss = GradientLoss()
edge_loss = SobelEdgeLoss().to(device)

fusion_head_optimizer = torch.optim.AdamW(fusion_head.parameters(), lr=0.0001)

fusion_head_scheduler = ReduceLROnPlateau(
    fusion_head_optimizer,
    mode='min',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=True
)

# Print current learning rate
current_lr2 = fusion_head_optimizer.param_groups[0]['lr']
print(f"Initial Learning Rate: {current_lr2:.6f}")

# Data directories
# Data directories
gt_dir = r"../high"
input_dir = r"../low"
output_dir = r"./lol"
test_dir = r"../test"
test_gt_dir = r"../test"
valid_input_dir = r"../val"
valid_gt_dir = r"../val_true"

# gt_dir = r"../collect/gt2"
# input_dir = r"../collect/input2"
# output_dir = r"./"
# valid_gt_dir = r"../collect/valid-gt2"
# valid_input_dir = r"../collect/valid-input2"
# test_dir = r"../collect/test"
# test_gt_dir = r"../collect/test"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load model weights if they exist
model_path = "model_weights-c.pth"
if os.path.exists(model_path) and args.phase == 'train':
    checkpoint = torch.load(model_path, map_location=device)
    fusion_head.load_state_dict(checkpoint['fusion_head_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    best_val_loss = 100
else:
    start_epoch = 0
    losses = []
    best_val_loss = float('inf')

# Training parameters
num_epochs = 10000
update_interval = 60
# Create datasets
if args.phase == 'train':
    train_dataset = ImageDataset(input_dir=input_dir, gt_dir=gt_dir, transform=None)
    val_dataset = ImageDataset(input_dir=valid_input_dir, gt_dir=valid_gt_dir, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # val_dataset = ImageDataset(input_dir=test_dir, gt_dir=test_gt_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    test_dataset = ImageDatasetTest(input_dir=test_dir, gt_dir=test_gt_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


def validate(model2, dataloader, device):
    model2.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (rgb_imgs, gt_imgs, _) in enumerate(dataloader):
            rgb_imgs = rgb_imgs.to(device).permute(0, 3, 1, 2) / 255.0
            gt_imgs = gt_imgs.to(device).permute(0, 3, 1, 2) / 255.0

            r_in = to_quaternion(rgb_imgs)
            r_gt = to_quaternion(gt_imgs)

            #
            # r_out2, _ = model(r_gt)
            #
            # l = l1_loss(r_out, r_out2)

            img = r_in

            test_data = {'ir': img, 'vis': img}
            fds = []
            diffusion.feed_data(test_data)
            for t in opt['model_df']['t']:
                _, fd_t = diffusion.get_feats(t=t)
                fds.append(fd_t)

            r_out = model2(fds, r_in)

            img = r_out[:, 1:, :, :]
            img_gt = r_gt[:, 1:, :, :]
            wavelet_low_l1 = mse_loss(img, img_gt)
            # wavelet_low_ssim = ll_subband_ssim_loss(img, img_gt)

            wavelet_low_cosine_loss = color_loss(img, img_gt)
            gradient_loss1 = gradient_loss(img, img_gt)
            # loss_tv = tv_loss(img)

            vgg_loss1 = vgg_loss(img, img_gt) * 0.1

            val_loss = wavelet_low_l1 + gradient_loss1 + wavelet_low_cosine_loss 

            total_loss += val_loss.item()

    return total_loss / len(dataloader)


log_file = open('training_log.txt', 'a')


def log_print(message, log_file='training_log.txt'):
    print(message)  # 输出到控制台
    with open(log_file, 'a') as f:  # 追加写入文件
        f.write(message + '\n')


if args.phase == 'train':
    # 创建测试子目录
    test_samples_dir = os.path.join(output_dir, "training_samples")
    os.makedirs(test_samples_dir, exist_ok=True)
    torch.cuda.empty_cache()
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # 按需获取测试样本
        test_sample = next(iter(test_loader))
        fusion_head.train()
        running_loss = 0.0
        #
        # Training phase
        for batch_idx, (rgb_imgs, gt_imgs, _) in enumerate(train_loader):
            rgb_imgs = rgb_imgs.to(device).permute(0, 3, 1, 2) / 255.0
            gt_imgs = gt_imgs.to(device).permute(0, 3, 1, 2) / 255.0

            r_in = to_quaternion(rgb_imgs)
            r_gt = to_quaternion(gt_imgs)

            img = r_in

            with torch.no_grad():  # 确保扩散模型不计算梯度
                test_data = {'ir': img, 'vis': img}
                fds = []
                diffusion.feed_data(test_data)
                for t in opt['model_df']['t']:
                    _, fd_t = diffusion.get_feats(t=t)
                    fds.append(fd_t)
                del test_data

            r_out = fusion_head(fds, r_in)

            img = r_out[:, 1:, :, :]
            img_gt = r_gt[:, 1:, :, :]
            wavelet_low_mse = mse_loss(img, img_gt)
            wavelet_low_l1 = l1_loss(img, img_gt)
            # wavelet_low_ssim = ll_subband_ssim_loss(img, img_gt)

            wavelet_low_cosine_loss = color_loss(img, img_gt)
            gradient_loss1 = gradient_loss(img, img_gt)
            loss_tv = tv_loss(img) * 0.1

            vgg_loss1 = vgg_loss(img, img_gt) * 0.1
            loss = wavelet_low_mse + gradient_loss1 + wavelet_low_cosine_loss 
            # 分别清零梯度
            fusion_head_optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 分别更新参数
            fusion_head_optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % update_interval == 0:
                avg_loss = running_loss / update_interval

                # Create the log message
                log_message = (
                    f'Epoch [{num_epochs}], Batch {batch_idx + 1}, '
                    f'Loss: {avg_loss:.6f}, '

                    f'wavelet_low_mse: {wavelet_low_mse:.6f}, '
                    # f'wavelet_low_ssim: {wavelet_low_ssim:.6f}, '
                    f'gradient_loss1: {gradient_loss1:.6f}, '
                    f'wavelet_low_cosine_loss: {wavelet_low_cosine_loss * 0.1:.6f}, '
                    # f'loss_tv: {loss_tv:.6f}, '
                )

                # Print to console
                print(log_message)

                # Write to file
                log_file.write(log_message + '\n')
                log_file.flush()  # Ensure the message is written immediately
                running_loss = 0.0
        # Validation phase
        val_loss = validate(fusion_head, val_loader, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.6f}')

        fusion_head_scheduler.step(val_loss)

        current_lr2 = fusion_head_optimizer.param_groups[0]['lr']
        log_print(f"Epoch [{epoch + 1}/{num_epochs}], Current Learning Rate: {current_lr2}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'fusion_head_state_dict': fusion_head.state_dict(),
                'fusion_head_optimizer_state_dict': fusion_head_optimizer.state_dict(),
                'losses': losses,
                'val_loss': val_loss
            }, 'best_model-lol.pth')
            log_print(f"Saved new best model with val loss------------------------------------: {val_loss:.6f}")

        # Periodic saving
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch + 1,
                'fusion_head_state_dict': fusion_head.state_dict(),
                'fusion_head_optimizer_state_dict': fusion_head_optimizer.state_dict(),
                'losses': losses,
                'val_loss': val_loss
            }, model_path)
            print(f'Saved model weights at epoch {epoch + 1}')

        # 每1个epoch测试一张样本 - 使用滑动窗口推理
        if (epoch + 1) % 1 == 0 and test_sample is not None:
            rgb_imgs, gt_imgs, filenames = test_sample
            # 开始计时
            start_time = time.time()
            # 将图像转换为PIL图像，然后转换为张量
            rgb_pil = Image.fromarray(rgb_imgs[0].numpy().astype('uint8'))
            gt_pil = Image.fromarray(gt_imgs[0].numpy().astype('uint8'))

            # 转换为张量并归一化
            input_tensor = to_tensor(rgb_pil).unsqueeze(0).to(device)  # [1, 3, H, W]
            gt_tensor = to_tensor(gt_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

            # 获取原始尺寸
            orig_h, orig_w = input_tensor.shape[2], input_tensor.shape[3]

            # 定义窗口大小和步长 (使用训练时的尺寸)
            window_size = (160, 160)  # (H, W)
            stride = (100, 100)  # (H, W)

            # 计算填充量
            pad_h = (stride[0] - (orig_h - window_size[0]) % stride[0]) % stride[0]
            pad_w = (stride[1] - (orig_w - window_size[1]) % stride[1]) % stride[1]

            # 使用反射填充
            padded_input = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            padded_gt = F.pad(gt_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            _, _, padded_h, padded_w = padded_input.shape

            # 初始化输出和权重矩阵
            output_padded = torch.zeros_like(padded_input)
            weight_mask = torch.zeros((1, 1, padded_h, padded_w), device=device)

            # 创建权重图（双线性权重）
            h_win, w_win = window_size
            weight_map = torch.ones(1, 1, h_win, w_win, device=device)

            center_y, center_x = h_win // 2, w_win // 2
            for y in range(h_win):
                for x in range(w_win):
                    # 计算到中心的距离并归一化
                    dist_y = min(y / center_y, (h_win - 1 - y) / center_y)
                    dist_x = min(x / center_x, (w_win - 1 - x) / center_x)
                    # 双线性权重
                    weight_map[0, 0, y, x] = dist_y * dist_x

            # 设置模型为评估模式
            fusion_head.eval()

            # 滑动窗口循环
            with torch.no_grad():
                for y in range(0, padded_h - h_win + 1, stride[0]):
                    for x in range(0, padded_w - w_win + 1, stride[1]):
                        # 提取当前窗口
                        patch = padded_input[:, :, y:y + h_win, x:x + w_win]

                        # 转换为四元数
                        r_in = to_quaternion(patch)
                        img_patch = r_in

                        # 通过扩散模型获取特征
                        test_data = {'ir': img_patch, 'vis': img_patch}
                        fds = []
                        diffusion.feed_data(test_data)
                        for t in opt['model_df']['t']:
                            _, fd_t = diffusion.get_feats(t=t)
                            fds.append(fd_t)

                        # 通过融合头得到输出
                        r_out = fusion_head(fds, r_in)
                        output_patch = r_out[:, 1:, :, :]  # 得到最终的输出图像块

                        # 将输出乘权重累加到输出矩阵
                        output_padded[:, :, y:y + h_win, x:x + w_win] += output_patch * weight_map
                        # 将权重累加到权重矩阵
                        weight_mask[:, :, y:y + h_win, x:x + w_win] += weight_map

                # 加权平均并裁剪
                output_padded /= weight_mask
                output_tensor = output_padded[:, :, :orig_h, :orig_w]
                end_time = time.time()
                inference_time = end_time - start_time

                # 打印推理时间
                print(f"Inference time for one image: {inference_time:.4f} seconds")

                # 如果需要，还可以计算FPS（每秒处理的图像数）
                fps = 1.0 / inference_time
                print(f"FPS: {fps:.2f}")

                # 转换为numpy图像并保存
                output_np = output_tensor.squeeze(0).cpu().clamp_(0, 1).numpy()
                output_np = (output_np.transpose(1, 2, 0) * 255).astype(np.uint8)

                # 保存结果
                filename = os.path.join(test_samples_dir, f"epoch_{epoch + 1}_sample.jpg")
                cv2.imwrite(filename, cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR))
                print(f"Saved sample test result to {filename}")


