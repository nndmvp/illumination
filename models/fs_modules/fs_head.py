import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Conv1d, Sigmoid
from pytorch_wavelets import DWTForward, DWTInverse


class QuaternionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(QuaternionConv1d, self).__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be divisible by 4 for quaternion."

        self.in_channels = in_channels // 4  # 四元数的四个分量 (r, i, j, k)
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # 合并权重为单个张量 (4, out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(torch.Tensor(4, self.out_channels, self.in_channels, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming初始化 + 权重归一化（可选）
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        with torch.no_grad():
            norm = self.weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
            self.weight.div_(norm)

    def forward(self, x):
        # 拆分输入为四元数组件 (batch, in_channels, L) -> 4 x (batch, in_channels//4, L)
        x_r, x_i, x_j, x_k = torch.chunk(x, 4, dim=1)

        # 拆分权重
        w_r, w_i, w_j, w_k = self.weight  # 4 x (out_channels, in_channels, kernel_size)

        # 构造拼接后的权重 (4*out_channels, 4*in_channels, kernel_size)
        cat_kernels_r = torch.cat([w_r, -w_i, -w_j, -w_k], dim=1)  # 实部计算
        cat_kernels_i = torch.cat([w_i, w_r, -w_k, w_j], dim=1)  # 虚部i计算
        cat_kernels_j = torch.cat([w_j, w_k, w_r, -w_i], dim=1)  # 虚部j计算
        cat_kernels_k = torch.cat([w_k, -w_j, w_i, w_r], dim=1)  # 虚部k计算
        cat_kernels = torch.cat([cat_kernels_r, cat_kernels_i, cat_kernels_j, cat_kernels_k], dim=0)

        # 拼接输入 (batch, 4*in_channels, L)
        cat_input = torch.cat([x_r, x_i, x_j, x_k], dim=1)

        # 单次卷积计算
        out = F.conv1d(cat_input, cat_kernels, bias=self.bias, stride=self.stride, padding=self.padding)

        return out


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:  # 256 x 256
            in_channels += inner_channel * channel_multiplier[0]
        elif scale < 6:  # 128 x 128
            in_channels += inner_channel * channel_multiplier[1]
        elif scale < 9:  # 64 x 64
            in_channels += inner_channel * channel_multiplier[2]
        elif scale < 12:  # 32 x 32
            in_channels += inner_channel * channel_multiplier[3]
        elif scale < 15:  # 16 x 16
            in_channels += inner_channel * channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14')
    return in_channels


class QuaternionEfficientHybridAttention(nn.Module):
    def __init__(self, in_planes, reduction=4, kernel_size=7):
        """
        高效四元数混合注意力模块（带条件输入）
        结合轻量级通道注意力和空间注意力设计

        :param in_planes: 输入通道数（四元数表示前的通道数）
        :param reduction: 通道压缩比例
        :param kernel_size: 空间注意力卷积核大小
        """
        super().__init__()
        self.in_planes = in_planes
        self.reduction = reduction
        self.kernel_size = kernel_size

        # 确保输入通道数是4的倍数（四元数要求）
        if in_planes % 4 != 0:
            raise ValueError("Input channels must be divisible by 4 for quaternion attention.")

        # ===== 条件调制模块 =====
        self.cond_pool = nn.AdaptiveAvgPool2d(1)  # 全局池化对齐维度
        self.cond_fc = nn.Sequential(
            nn.Linear(4, in_planes * 2),  # 生成gamma和beta
            nn.GELU()
        )

        # ===== 通道注意力分支 =====
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            QuaternionConv2d(in_planes, in_planes // reduction, kernel_size=1),
            QuaternionLeakyReLU(negative_slope=0.2),
            QuaternionConv2d(in_planes // reduction, in_planes, kernel_size=1),
            QuaternionLeakyReLU(negative_slope=0.2)
        )

        # ===== 空间注意力分支 =====
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.LeakyReLU(0.1)
        )

        # ===== 轻量级特征增强 =====
        self.depthwise = QuaternionConv2d(
            in_planes, in_planes, kernel_size=3,
            padding=1
        )
        self.pointwise = QuaternionConv2d(in_planes, in_planes, kernel_size=1)

    def forward(self, x):
        """
        :param x: 输入特征 [B, 4*C, H, W]（四元数表示）
        :param img: 条件图像 [B, 4, H_img, W_img]
        :return: 注意力加权后的特征 [B, 4*C, H, W]
        """
        # 保存原始输入用于残差连接
        identity = x

        # # 1. 条件特征调制
        # cond = self.cond_pool(img)  # [B, 4, 1, 1]
        # cond = cond.view(cond.size(0), -1)  # [B, 4]
        # gamma, beta = self.cond_fc(cond).chunk(2, dim=1)  # 拆分为调制参数
        # gamma = gamma.view(-1, self.in_planes, 1, 1)  # [B, 4*C, 1, 1]
        # beta = beta.view(-1, self.in_planes, 1, 1)  # [B, 4*C, 1, 1]

        # 2. 轻量级特征增强 + 条件调制
        x = self.depthwise(x)
        x = self.pointwise(x)
        # x = x * (1 + gamma) + beta  # 条件调制

        # 3. 通道注意力
        channel_att = self.channel_att(x)  # [B, 4*C, 1, 1]

        # 4. 空间注意力
        # 计算四元数模长作为空间注意力基础
        r, i, j, k = torch.chunk(x, 4, dim=1)
        magnitude = torch.sqrt(r ** 2 + i ** 2 + j ** 2 + k ** 2 + 1e-8)  # [B, C, H, W]

        # 通道维度的平均和最大池化
        avg_out = torch.mean(magnitude, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(magnitude, dim=1, keepdim=True)  # [B, 1, H, W]

        # 拼接特征
        spatial_base = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial_att = self.spatial_att(spatial_base)  # [B, 1, H, W]

        # 将空间注意力扩展到四元数通道
        spatial_att = spatial_att.repeat(1, self.in_planes, 1, 1)  # [B, 4*C, H, W]

        # 5. 组合注意力
        combined_att = channel_att * spatial_att  # [B, 4*C, H, W]

        # 6. 应用注意力并残差连接
        return identity + x * combined_att


class QuaternionChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(QuaternionChannelAttention, self).__init__()
        self.in_planes = in_planes // 4  # 四元数的通道数是输入的1/4
        self.ratio = ratio

        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 四元数卷积层
        self.fc1 = QuaternionConv2d(self.in_planes * 4, self.in_planes * 4 // ratio, kernel_size=1)
        self.relu1 = QuaternionLeakyReLU(negative_slope=0.2)
        self.fc2 = QuaternionConv2d(self.in_planes * 4 // ratio, self.in_planes * 4, kernel_size=1)

        # Sigmoid 激活函数
        self.sigmoid = QuaternionLeakyReLU(0.2)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        y = self.avg_pool(x)
        y = self.fc1(y)
        # 全局平均池化
        avg_out = self.fc2(self.relu1(y))
        # 全局最大池化
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        # 合并平均池化和最大池化的结果
        out = avg_out + max_out
        out = self.sigmoid(out)

        # 将注意力权重扩展为与输入 x 相同的形状
        out = out.expand_as(x)
        return out


class QuaternionSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, channels=64):
        super(QuaternionSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 四元数卷积层
        self.conv1 = QuaternionConv2d(8, channels, kernel_size, padding=padding)  # 输出通道数与输入一致
        self.sigmoid = QuaternionLeakyReLU(0.2)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 检查输入通道数是否是4的倍数（四元数要求）
        if channels % 4 != 0:
            raise ValueError("Input channels must be divisible by 4 for quaternion spatial attention.")

        # 计算通道维度的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, height, width]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [batch_size, 1, height, width]

        # 将平均值和最大值分别扩展为四元数的四个分量
        avg_out = torch.cat([avg_out, avg_out, avg_out, avg_out], dim=1)  # [batch_size, 4, height, width]
        max_out = torch.cat([max_out, max_out, max_out, max_out], dim=1)  # [batch_size, 4, height, width]

        # 合并平均值和最大值
        x = torch.cat([avg_out, max_out], dim=1)  # [batch_size, 8, height, width]

        # 四元数卷积
        x = self.conv1(x)  # [batch_size, channels, height, width]

        # 使用 Sigmoid 激活函数
        return self.sigmoid(x)


class QuaternionLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.1, inplace=False):
        super(QuaternionLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        # 检查输入通道数是否是4的倍数（四元数要求）
        if x.size(1) % 4 != 0:
            raise ValueError("Input channels must be divisible by 4 for quaternion LeakyReLU.")

        # 拆分四元数为四个分量 (r, i, j, k)
        chunks = torch.chunk(x, 4, dim=1)

        # 对每个分量分别应用 LeakyReLU
        activated_chunks = []
        for chunk in chunks:
            activated_chunks.append(
                F.leaky_relu(chunk,
                             negative_slope=self.negative_slope,
                             inplace=self.inplace)
            )

        # 合并处理后的分量
        return torch.cat(activated_chunks, dim=1)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads

        # 确保维度是4的倍数（四元数要求）
        assert dim % 4 == 0, "Dimension must be divisible by 4 for quaternion attention"
        self.dim = dim
        self.quat_dim = dim // 4  # 四元数维度

        # 使用更小的初始温度值，防止注意力分数过大
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1)

        # 使用四元数卷积替代普通卷积
        self.qkv = QuaternionConv2d(dim, dim * 3, kernel_size=1, bias=bias)

        # 深度可分离卷积保持普通形式，但确保输入输出通道数是4的倍数
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )

        # 输出投影使用四元数卷积
        self.project_out = QuaternionConv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        # 可学习的注意力权重参数，使用更小的初始值
        self.attn1 = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        # 添加层归一化稳定训练
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape

        # 生成查询、键、值
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 应用层归一化
        q = self.norm_q(q.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        k = self.norm_k(k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        v = self.norm_v(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 重整形为多头形式
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 四元数归一化 - 对每个四元数进行归一化
        # q = self.quaternion_normalize(q)
        # k = self.quaternion_normalize(k)

        _, _, C, _ = q.shape

        # 创建注意力掩码
        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        # 计算注意力分数，添加数值稳定性
        attn = (q @ k.transpose(-2, -1)) * self.temperature.clamp(min=1e-8, max=1e4)

        # 创建四种不同稀疏度的注意力模式
        index = torch.topk(attn, k=max(1, int(C / 2)), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, -1e4))  # 使用有限值替代-inf

        index = torch.topk(attn, k=max(1, int(C * 2 / 3)), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, -1e4))

        index = torch.topk(attn, k=max(1, int(C * 3 / 4)), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, -1e4))

        index = torch.topk(attn, k=max(1, int(C * 4 / 5)), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, -1e4))

        # 应用softmax，添加数值稳定性
        attn1 = F.softmax(attn1.clamp(min=-1e4, max=1e4), dim=-1)
        attn2 = F.softmax(attn2.clamp(min=-1e4, max=1e4), dim=-1)
        attn3 = F.softmax(attn3.clamp(min=-1e4, max=1e4), dim=-1)
        attn4 = F.softmax(attn4.clamp(min=-1e4, max=1e4), dim=-1)

        # 计算四种不同的输出
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        # 加权融合四种不同的注意力输出
        out = out1 * self.attn1.clamp(min=0.01, max=1.0) + \
              out2 * self.attn2.clamp(min=0.01, max=1.0) + \
              out3 * self.attn3.clamp(min=0.01, max=1.0) + \
              out4 * self.attn4.clamp(min=0.01, max=1.0)

        # 重整形回原始格式
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 输出投影
        out = self.project_out(out)
        return out

    def quaternion_normalize(self, x):
        """四元数归一化 - 对每个四元数进行归一化"""
        b, heads, c, n = x.shape
        # 确保通道数是4的倍数
        assert c % 4 == 0, "Channels must be divisible by 4 for quaternion normalization"

        # 重塑为四元数形式 [b, heads, c//4, 4, n]
        x_reshaped = x.view(b, heads, c // 4, 4, n)

        # 计算每个四元数的模长
        modulus = torch.sqrt(torch.sum(x_reshaped ** 2, dim=3, keepdim=True) + 1e-8)

        # 归一化
        normalized = x_reshaped / modulus

        # 恢复原始形状
        return normalized.view(b, heads, c, n)


class QuaternionEfficientHybridAttentionSVD(nn.Module):
    def __init__(self, in_planes, reduction=4, kernel_size=7, rank_ratio=0.25):
        """
        使用SVD低秩近似的四元数高效混合注意力模块

        :param in_planes: 输入通道数
        :param reduction: 通道压缩比例
        :param kernel_size: 空间注意力卷积核大小
        :param rank_ratio: 低秩近似的比例 (0-1)
        """
        super().__init__()
        self.in_planes = in_planes
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.rank_ratio = rank_ratio

        # 计算低秩近似的秩
        self.rank = max(1, int(in_planes * rank_ratio))

        if in_planes % 4 != 0:
            raise ValueError("Input channels must be divisible by 4 for quaternion attention.")

        # ===== 通道注意力分支 (使用SVD低秩近似) =====
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # 使用SVD低秩近似的卷积替代标准卷积
            SVDQuaternionConv2d(in_planes, in_planes // reduction, kernel_size=1, rank_ratio=rank_ratio),
            QuaternionLeakyReLU(negative_slope=0.2),
            SVDQuaternionConv2d(in_planes // reduction, in_planes, kernel_size=1, rank_ratio=rank_ratio),
            QuaternionLeakyReLU(negative_slope=0.2)
        )

        # ===== 空间注意力分支 =====
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.LeakyReLU(0.1)
        )

        # ===== 轻量级特征增强 (使用SVD低秩近似) =====
        self.depthwise = SVDQuaternionConv2d(
            in_planes, in_planes, kernel_size=3,
            padding=1, rank_ratio=rank_ratio
        )
        self.pointwise = SVDQuaternionConv2d(in_planes, in_planes, kernel_size=1, rank_ratio=rank_ratio)

    def forward(self, x):
        identity = x

        # 1. 轻量级特征增强 (使用SVD低秩近似)
        x = self.depthwise(x)
        x = self.pointwise(x)

        # 2. 通道注意力 (使用SVD低秩近似)
        channel_att = self.channel_att(x)

        # 3. 空间注意力
        r, i, j, k = torch.chunk(x, 4, dim=1)
        magnitude = torch.sqrt(r ** 2 + i ** 2 + j ** 2 + k ** 2 + 1e-8)

        avg_out = torch.mean(magnitude, dim=1, keepdim=True)
        max_out, _ = torch.max(magnitude, dim=1, keepdim=True)

        spatial_base = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_att(spatial_base)
        spatial_att = spatial_att.repeat(1, self.in_planes, 1, 1)

        # 4. 组合注意力
        combined_att = channel_att * spatial_att

        # 5. 应用注意力并残差连接
        return identity + x * combined_att


class SVDQuaternionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rank_ratio=0.25):
        """
        使用SVD低秩近似的四元数卷积层

        :param rank_ratio: 低秩近似的比例
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank_ratio = rank_ratio
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 计算低秩近似的秩
        self.rank = max(1, int(min(in_channels, out_channels) * rank_ratio))

        # 创建低秩分解的权重矩阵
        self.U = nn.Parameter(torch.randn(out_channels, self.rank))
        self.V = nn.Parameter(torch.randn(self.rank, in_channels * kernel_size * kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.U)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 重建权重矩阵
        weight = torch.mm(self.U, self.V)
        weight = weight.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        # 应用卷积
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuaternionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(QuaternionConv2d, self).__init__()
        self.in_channels = in_channels // 4  # 四元数的四个分量 (r, i, j, k)
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding

        # 初始化四元数权重 (r, i, j, k)
        self.weight = nn.Parameter(torch.Tensor(4, self.out_channels, self.in_channels, kernel_size, kernel_size))

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

        # 拆分权重
        w_r, w_i, w_j, w_k = self.weight  # 4 x (out_channels, in_channels, kH, kW)

        # 构造拼接后的权重 (4*out_channels, in_channels, kH, kW)
        cat_kernels_r = torch.cat([w_r, -w_i, -w_j, -w_k], dim=1)  # 对应实部计算
        cat_kernels_i = torch.cat([w_i, w_r, -w_k, w_j], dim=1)  # 对应虚部i计算
        cat_kernels_j = torch.cat([w_j, w_k, w_r, -w_i], dim=1)  # 对应虚部j计算
        cat_kernels_k = torch.cat([w_k, -w_j, w_i, w_r], dim=1)  # 对应虚部k计算

        # 最终拼接 (4*out_channels, 4*in_channels, kH, kW)
        cat_kernels = torch.cat([cat_kernels_r, cat_kernels_i, cat_kernels_j, cat_kernels_k], dim=0)

        # 拼接输入 (batch, 4*in_channels, H, W)
        cat_input = torch.cat([x_r, x_i, x_j, x_k], dim=1)

        # 单次卷积计算
        out = F.conv2d(cat_input, cat_kernels, bias=self.bias, stride=self.stride, padding=self.padding)

        return out


class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            QuaternionConv2d(dim * len(time_steps), dim * len(time_steps), kernel_size=3, padding=1),
            QuaternionBatchNorm2d(dim * len(time_steps)),
            QuaternionLeakyReLU(0.2)
        )

        self.conv = QuaternionConv2d(dim * len(time_steps), dim_out, kernel_size=3, padding=1)
        self.norm = QuaternionBatchNorm2d(dim_out)
        self.fusion_activation = QuaternionLeakyReLU(negative_slope=0.2)

    def forward(self, x):
        r = self.block(x)

        x = x + r
        x = self.conv(x)
        x = self.fusion_activation(x)
        return x


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
                QuaternionConv2d(channel // reduction, channel, kernel_size=1),
                QuaternionLeakyReLU(0.2)
            )

        # 高效通道注意力模块
        # 注意：输入通道数为1，因为我们处理的是标量权重
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)

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


class HeadLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadLeakyRelu2d, self).__init__()
        self.conv = QuaternionConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.leaky = QuaternionLeakyReLU(0.2)

    def forward(self, x):
        return self.leaky(self.conv(x))


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


from torch import einsum
from einops import rearrange, reduce


class DynamicFeatureInteraction(nn.Module):
    """
    动态特征交互模块 - 使用门控机制和动态权重学习
    让每个特征都能选择性地关注其他特征
    """

    def __init__(self, in_channels, reduction_ratio=8, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = max(4, in_channels // num_heads)
        self.hidden_dim = max(8, in_channels // reduction_ratio)

        # 动态权重生成网络
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, self.hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, in_channels * in_channels, 1),
            nn.Sigmoid()
        )

        # 门控机制
        self.gate = nn.Sequential(
            QuaternionConv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

        # 归一化
        self.norm = QuaternionBatchNorm2d(in_channels)  # 假设in_channels是4的倍数

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # 生成动态权重矩阵
        weight_matrix = self.weight_generator(x).view(batch_size, channels, channels)

        # 重塑输入以进行矩阵乘法
        x_flat = x.view(batch_size, channels, -1)

        # 应用动态权重 - 每个特征与其他特征的加权组合
        weighted_features = torch.bmm(weight_matrix, x_flat)
        weighted_features = weighted_features.view(batch_size, channels, height, width)

        # 门控机制 - 决定保留多少原始信息
        gate_input = torch.cat([x, weighted_features], dim=1)
        gate_value = self.gate(gate_input)

        # 输出 = 门控值 * 加权特征 + (1 - 门控值) * 原始输入
        out = gate_value * weighted_features + (1 - gate_value) * x

        return self.norm(out)


class LinearAttention(nn.Module):
    """
    线性注意力机制 - 内存高效
    基于"Transformers are RNNs"论文中的线性注意力
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        # 线性注意力计算
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        return self.to_out(out)


class EfficientGlobalAttention(nn.Module):
    """
    高效全局注意力模块
    结合动态特征交互和线性注意力
    """

    def __init__(self, in_channels, reduction_ratio=8, num_heads=4):
        super().__init__()
        self.in_channels = in_channels

        # 动态特征交互
        self.dynamic_interaction = DynamicFeatureInteraction(
            in_channels, reduction_ratio, num_heads
        )

        # 线性注意力
        self.linear_attention = LinearAttention(in_channels, heads=num_heads)

        # 自适应权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 动态特征交互
        dynamic_out = self.dynamic_interaction(x)

        # 线性注意力
        attention_out = self.linear_attention(x)

        # 自适应融合
        out = self.alpha * dynamic_out + self.beta * attention_out

        return out + x  # 残差连接


class DifferentiableLSH(nn.Module):
    """
    可微分局部敏感哈希(LSH)模块
    直接处理高维特征输入，保持原始通道数
    """

    def __init__(self, in_channels=16, nbin=16, sigma=10.0, alpha_scale=0.1):
        super(DifferentiableLSH, self).__init__()
        self.in_channels = in_channels
        self.nbin = nbin
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.alpha_scale = alpha_scale

        # 可学习的特征区间中心
        self.feature_centers = nn.Parameter(torch.linspace(-1, 1, nbin))

        # 条件网络 - 生成条件映射
        self.conditional_net = nn.Sequential(
            QuaternionConv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            QuaternionConv2d(in_channels * 2, in_channels * nbin, kernel_size=1),
            nn.Sigmoid()  # 输出值在[0,1]范围内
        )

        # 可学习的双向递归滤波器
        # 使用分组卷积来处理高维特征
        self.horizontal_forward = nn.Conv2d(nbin * in_channels, nbin * in_channels,
                                            kernel_size=(1, 3), padding=(0, 1),
                                            groups=nbin * in_channels, bias=False)
        self.horizontal_backward = nn.Conv2d(nbin * in_channels, nbin * in_channels,
                                             kernel_size=(1, 3), padding=(0, 1),
                                             groups=nbin * in_channels, bias=False)
        self.vertical_forward = nn.Conv2d(nbin * in_channels, nbin * in_channels,
                                          kernel_size=(3, 1), padding=(1, 0),
                                          groups=nbin * in_channels, bias=False)
        self.vertical_backward = nn.Conv2d(nbin * in_channels, nbin * in_channels,
                                           kernel_size=(3, 1), padding=(1, 0),
                                           groups=nbin * in_channels, bias=False)

        # 自适应权重生成网络
        self.adaptive_net = nn.Sequential(
            QuaternionConv2d(nbin * in_channels, in_channels // 4, kernel_size=1),
            QuaternionLeakyReLU(0.2),
            QuaternionBatchNorm2d(in_channels // 4),
            QuaternionConv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 初始化滤波器权重
        self._initialize_filters()

    def _initialize_filters(self):
        # 初始化双向递归滤波器
        with torch.no_grad():
            # 前向滤波器权重
            alpha = math.exp(-math.sqrt(2) / (self.sigma.item() * 256))
            forward_weights = torch.tensor([0, 1.0, alpha]).view(1, 1, 1, 3)
            forward_weights = forward_weights / forward_weights.sum()

            # 后向滤波器权重（反向顺序）
            backward_weights = torch.tensor([alpha, 1.0, 0]).view(1, 1, 1, 3)
            backward_weights = backward_weights / backward_weights.sum()

            # 设置滤波器权重
            for i in range(self.nbin * self.in_channels):
                self.horizontal_forward.weight[i] = forward_weights
                self.horizontal_backward.weight[i] = backward_weights
                self.vertical_forward.weight[i] = forward_weights.permute(0, 1, 3, 2)
                self.vertical_backward.weight[i] = backward_weights.permute(0, 1, 3, 2)

    def _apply_bidirectional_filter(self, x, forward_conv, backward_conv, flip_dim):
        # 前向传递
        forward = forward_conv(x)

        # 后向传递（需要翻转输入）
        flipped = torch.flip(x, [flip_dim])
        backward_flipped = backward_conv(flipped)
        backward = torch.flip(backward_flipped, [flip_dim])

        # 合并前向和后向结果（减去原始值避免重复计算）
        return forward + backward - x

    def forward(self, x):
        """
        前向传播 - 直接处理高维特征
        x: 输入张量，形状为 [B, C, H, W] (任意通道数)
        返回: LSH处理后的特征，形状为 [B, C, H, W]
        """
        # 保存原始输入用于残差连接
        identity = x

        # 获取输入形状
        b, c, h, w = x.shape

        # 计算条件映射
        conditional_map = self.conditional_net(x)  # 输出形状: [B, C*nbin, H, W]

        # 将特征中心重塑为 [1, 1, nbin, 1, 1] 用于广播
        feature_centers = self.feature_centers.view(1, 1, -1, 1, 1)
        # 扩展输入维度为 [B, C, 1, H, W] 用于与特征中心比较
        x_expanded = x.unsqueeze(2)

        # 将条件映射重塑为与特征差值相同的形状 [B, C, nbin, H, W]
        conditional_map_reshaped = conditional_map.view(b, c, self.nbin, h, w)

        # 计算特征值与区间中心的差值 [B, C, nbin, H, W]
        feature_diff = x_expanded - feature_centers

        # 使用条件映射调整特征差值，计算软分配权重
        soft_assign = feature_diff * conditional_map_reshaped

        # 对软分配权重进行归一化，确保每个位置的所有区间权重和为1
        soft_assign = torch.softmax(soft_assign, dim=2)

        # 重塑为 [B, nbin*C, H, W]
        soft_assign_reshaped = soft_assign.view(b, -1, h, w)

        # 应用双向水平滤波（在宽度方向，翻转维度3）
        hist_h = self._apply_bidirectional_filter(
            soft_assign_reshaped,
            self.horizontal_forward,
            self.horizontal_backward,
            flip_dim=3
        )

        # 应用双向垂直滤波（在高度方向，翻转维度2）
        hist = self._apply_bidirectional_filter(
            hist_h,
            self.vertical_forward,
            self.vertical_backward,
            flip_dim=2
        )
        # 计算自适应权重

        # 使用软分配权重和直方图值计算输出
        # 重塑直方图为 [B, C, nbin, H, W]
        hist_reshaped = hist.view(b, c, self.nbin, h, w)

        # 加权求和（可微分替代硬索引）
        output = hist * soft_assign_reshaped

        # 应用自适应权重
        output = self.adaptive_net(output)

        # 残差连接
        return output + identity


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


class Fusion_Head(nn.Module):
    '''
    Change detection head (version 2).
    '''

    def __init__(self, feat_scales, out_channels=3, inner_channel=None, channel_multiplier=None, img_size=256,
                 time_steps=None):
        super(Fusion_Head, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.in_channels = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size = img_size
        self.time_steps = time_steps

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        self.decoder2 = nn.ModuleList()

        for i in range(0, len(self.feat_scales)):
            dim = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )
            self.decoder2.append(
                AttentionBlock(dim)
                # EnhancedConditionalProcessing(dim, dim)
            )
            if i != len(self.feat_scales) - 1:
                dim_out = get_in_channels([self.feat_scales[i + 1]], inner_channel, channel_multiplier)

                self.decoder.append(
                    EnhancedConditionalProcessing(dim, dim_out)
                )
                self.decoder2.append(
                    # LowRankGlobalAttention(dim_out)
                    DifferentiableLSH(dim_out)
                )

        # Convolutional layers before parsing to difference head

        self.init = EnhancedConditionalProcessing(4,
                                                  get_in_channels([self.feat_scales[len(self.feat_scales) - 1]],
                                                                  inner_channel, channel_multiplier))
        self.d = nn.ModuleList()

        for i in reversed(range(0, len(self.feat_scales))):
            dim = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)
            if i != 0:
                dim_out = get_in_channels([self.feat_scales[i - 1]], inner_channel, channel_multiplier)
                self.d.append(
                    EnhancedConditionalProcessing(dim, dim_out)
                )

        self.enhanced = EnhancedConditionalProcessing(64, 16)
        self.rank = DifferentiableLSH(16)
        # self.attention = AttentionBlock(64, 64)

        # Final head
        self.rgb_decode2 = HeadLeakyRelu2d(16, 16)
        self.rgb_decode1 = QuaternionConv2d(16, 4, kernel_size=3, padding=1)

        self.leaky = nn.Sigmoid()
        self.dropout = QuaternionDropout(0.2)

    def forward(self, feats, img):
        # Decoder
        lvl = 0
        img_t = 0

        lgl = len(self.feat_scales) - 2
        intermediate_ims = []
        im = self.init(img)
        intermediate_ims.append(im.clone())
        for ly in self.d:
            size = feats[0][self.feat_scales[lgl]].size()
            im = F.interpolate(im, size=(size[2], size[3]), mode="bilinear", align_corners=True)
            lgl = lgl - 1

            im = ly(im)
            intermediate_ims.append(im.clone())

        l = len(intermediate_ims) - 1

        for layer, layer2 in zip(self.decoder, self.decoder2):
            if isinstance(layer, Block):
                f_s = feats[0][self.feat_scales[lvl]]  # feature stacked
                if len(self.time_steps) > 1:
                    for i in range(1, len(self.time_steps)):
                        f_s = torch.cat((f_s, feats[i][self.feat_scales[lvl]]), dim=1)
                    f_s = layer(f_s)
                # if lvl != 0:
                #     f_s = f_s + x
                ims = intermediate_ims[l]

                # f_s = torch.cat((f_s, intermediate_ims[l]), dim=1)
                # f_s = f_s + intermediate_ims[l]
                if lvl != 0:
                    ims = ims + x

                f_s = layer2(f_s)

                f_s = f_s + ims
                # f_s = torch.cat((f_s, ims), dim=1)
                # f_s = intermediate_ims[l] - f_s
                # if lvl != 0:
                #     f_s = f_s + x
                lvl += 1
                l -= 1

            else:
                f_s = layer(f_s)
                f_s = layer2(f_s)
                size = feats[0][self.feat_scales[lvl]].size()
                x = F.interpolate(f_s, size=(size[2], size[3]), mode="bilinear", align_corners=True)

        # f_s = self.attention(f_s)
        f_s = self.enhanced(f_s)
        f_s = self.rank(f_s)
        # Fusion Head
        f_s = self.rgb_decode2(f_s)
        f_s = self.dropout(f_s)
        out = self.rgb_decode1(f_s)

        return self.leaky(out)
