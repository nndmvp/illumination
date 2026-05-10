import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sr3_modules.se import ChannelSpatialSELayer
import math
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Conv1d, Sigmoid
from pytorch_wavelets import DWTForward, DWTInverse


class QuaternionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(QuaternionConv1d, self).__init__()
        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.weight_r = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size))
        self.weight_i = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size))
        self.weight_j = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size))
        self.weight_k = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_r, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight_i, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight_j, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight_k, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x_r, x_i, x_j, x_k = torch.chunk(x, 4, dim=1)

        y_r = F.conv1d(x_r, self.weight_r, self.bias, self.stride, self.padding) - \
              F.conv1d(x_i, self.weight_i, None, self.stride, self.padding) - \
              F.conv1d(x_j, self.weight_j, None, self.stride, self.padding) - \
              F.conv1d(x_k, self.weight_k, None, self.stride, self.padding)

        y_i = F.conv1d(x_r, self.weight_i, None, self.stride, self.padding) + \
              F.conv1d(x_i, self.weight_r, None, self.stride, self.padding) + \
              F.conv1d(x_j, self.weight_k, None, self.stride, self.padding) - \
              F.conv1d(x_k, self.weight_j, None, self.stride, self.padding)

        y_j = F.conv1d(x_r, self.weight_j, None, self.stride, self.padding) - \
              F.conv1d(x_i, self.weight_k, None, self.stride, self.padding) + \
              F.conv1d(x_j, self.weight_r, None, self.stride, self.padding) + \
              F.conv1d(x_k, self.weight_i, None, self.stride, self.padding)

        y_k = F.conv1d(x_r, self.weight_k, None, self.stride, self.padding) + \
              F.conv1d(x_i, self.weight_j, None, self.stride, self.padding) - \
              F.conv1d(x_j, self.weight_i, None, self.stride, self.padding) + \
              F.conv1d(x_k, self.weight_r, None, self.stride, self.padding)

        y = torch.cat([y_r, y_i, y_j, y_k], dim=1)
        return y


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


class QuaternionChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(QuaternionChannelAttention, self).__init__()
        self.in_planes = in_planes // 4  # 四元数的通道数是输入的1/4
        self.ratio = ratio

        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 四元数卷积层
        self.fc1 = QuaternionConv2d2(self.in_planes * 4, self.in_planes * 4 // ratio, kernel_size=1)
        self.relu1 = QuaternionLeakyReLU(negative_slope=0.1)
        self.fc2 = QuaternionConv2d2(self.in_planes * 4 // ratio, self.in_planes * 4, kernel_size=1)

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


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = QuaternionChannelAttention(in_planes, ratio)
        self.spatial_attention = QuaternionSpatialAttention(kernel_size, channels=in_planes)

    def forward(self, x):
        # 通道注意力
        y = self.channel_attention(x)
        x = x * y  # 现在 x 和 y 的形状相同
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


class QuaternionLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.1, inplace=False):
        super(QuaternionLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 检查输入通道数是否是4的倍数（四元数要求）
        if channels % 4 != 0:
            raise ValueError("Input channels must be divisible by 4 for quaternion LeakyReLU.")

        # 将四元数拆分为实部和虚部
        x_real = x[:, :channels // 4, :, :]  # 实部
        x_imag = x[:, channels // 4:, :, :]  # 虚部

        # 对实部和虚部分别应用 LeakyReLU
        x_real = F.leaky_relu(x_real, negative_slope=self.negative_slope, inplace=self.inplace)
        x_imag = F.leaky_relu(x_imag, negative_slope=self.negative_slope, inplace=self.inplace)

        # 合并实部和虚部
        x_out = torch.cat([x_real, x_imag], dim=1)
        return x_out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = EnhancedQuaternionConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = QuaternionBatchNorm2d(out_channels)
        self.activation = QuaternionLeakyReLU(0.2)
        self.conv2 = EnhancedQuaternionConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = QuaternionBatchNorm2d(out_channels)
        self.se_layer = QuaternionSELayer(out_channels, reduction=8, cond_dim=0)
        self.conv3 = EnhancedQuaternionConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm3 = QuaternionBatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        residual = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.se_layer(out)
        out += residual

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.activation(out)
        return out


class QuaternionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(QuaternionConv2d, self).__init__()
        self.in_channels = in_channels // 4  # 因为每个四元数有四个部分
        self.out_channels = out_channels // 4
        self.padding = padding

        # 初始化四元数卷积的四个分量的权重
        self.r_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.i_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.j_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.k_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))

        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        # 使用适当的初始化策略初始化四元数权重
        nn.init.kaiming_uniform_(self.r_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.i_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.j_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))

    def forward(self, input):
        # 假设输入是形状为 (batch, in_channels, H, W) 的四元数数据
        batch_size, _, height, width = input.size()

        # 拆分输入为四元数组件
        x_r, x_i, x_j, x_k = torch.chunk(input, 4, dim=1)

        # 四元数卷积计算
        conv_rr = F.conv2d(x_r, self.r_weight, stride=1, padding=self.padding)
        conv_ri = F.conv2d(x_r, self.i_weight, stride=1, padding=self.padding)
        conv_rj = F.conv2d(x_r, self.j_weight, stride=1, padding=self.padding)
        conv_rk = F.conv2d(x_r, self.k_weight, stride=1, padding=self.padding)

        conv_ir = F.conv2d(x_i, self.r_weight, stride=1, padding=self.padding)
        conv_ii = F.conv2d(x_i, self.i_weight, stride=1, padding=self.padding)
        conv_ij = F.conv2d(x_i, self.j_weight, stride=1, padding=self.padding)
        conv_ik = F.conv2d(x_i, self.k_weight, stride=1, padding=self.padding)

        conv_jr = F.conv2d(x_j, self.r_weight, stride=1, padding=self.padding)
        conv_ji = F.conv2d(x_j, self.i_weight, stride=1, padding=self.padding)
        conv_jj = F.conv2d(x_j, self.j_weight, stride=1, padding=self.padding)
        conv_jk = F.conv2d(x_j, self.k_weight, stride=1, padding=self.padding)

        conv_kr = F.conv2d(x_k, self.r_weight, stride=1, padding=self.padding)
        conv_ki = F.conv2d(x_k, self.i_weight, stride=1, padding=self.padding)
        conv_kj = F.conv2d(x_k, self.j_weight, stride=1, padding=self.padding)
        conv_kk = F.conv2d(x_k, self.k_weight, stride=1, padding=self.padding)

        # 根据四元数乘法规则合并结果
        out_r = conv_rr - conv_ri - conv_rj - conv_rk
        out_i = conv_ir + conv_ii + conv_ij - conv_ik
        out_j = conv_jr - conv_ji + conv_jj + conv_jk
        out_k = conv_kr + conv_ki - conv_kj + conv_kk

        # 组合输出
        out = torch.cat([out_r, out_i, out_j, out_k], dim=1)
        return out


class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            EnhancedQuaternionConv2d(dim * len(time_steps), dim * len(time_steps), kernel_size=3, padding=1),
            QuaternionLeakyReLU(),
            EnhancedQuaternionConv2d(dim * len(time_steps), dim * len(time_steps), kernel_size=3, padding=1),
            QuaternionLeakyReLU()
        )
        self.se = QuaternionSELayer(dim * len(time_steps), reduction=16, cond_dim=0)

        self.conv = EnhancedQuaternionConv2d(dim * len(time_steps), dim_out, kernel_size=3, padding=1)
        self.fusion_activation = QuaternionLeakyReLU(negative_slope=0.1)

    def forward(self, x, cond):
        r = self.block(x)
        x = self.se(r)
        x = x + r
        x = self.conv(x)
        x = self.fusion_activation(x)
        return x


class EnhancedQuaternionBlock(nn.Module):
    def __init__(self, dim, dim_out, time_steps, reduction=16):
        super().__init__()

        # 输入预处理层
        self.input_proj = nn.Sequential(
            EnhancedQuaternionConv2d(dim * len(time_steps), dim, kernel_size=1),
            QuaternionBatchNorm2d(dim)  # 新增四元数层归一化
        )

        # 主干卷积路径
        self.conv_path = nn.Sequential(
            EnhancedQuaternionConv2d(dim, dim, kernel_size=3, padding=1, groups=4),  # 分组卷积提升效率
            QuaternionLeakyReLU(negative_slope=0.1),
            EnhancedQuaternionConv2d(dim, dim, kernel_size=3, padding=1, dilation=2),  # 空洞卷积扩大感受野
            QuaternionLeakyReLU(negative_slope=0.1)
        )

        # 条件注意力模块
        self.cond_attention = QuaternionSELayer(dim, reduction=16, cond_dim=12)

        # 残差连接处理
        self.res_conv = EnhancedQuaternionConv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

        # 输出处理
        self.output = nn.Sequential(
            QuaternionDropout(0.1),  # 新增正则化
            EnhancedQuaternionConv2d(dim, dim_out, kernel_size=3, padding=1),
            QuaternionLeakyReLU(negative_slope=0.1)
        )

    def forward(self, x, cond):
        # 输入预处理
        x = self.input_proj(x)

        # 保存残差
        identity = x

        # 主干卷积
        x = self.conv_path(x)

        # 条件注意力
        x = self.cond_attention(x, cond)

        # 残差连接
        x = x + identity

        # 输出转换
        x = self.res_conv(x)
        x = self.output(x)

        return x


class QuaternionDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        四元数Dropout层
        Args:
            p (float): 丢弃概率，默认0.5
        """
        super().__init__()
        self.p = p
        self.bernoulli = torch.distributions.Bernoulli(probs=1 - p)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # 生成共享的mask [B,1,H,W]（四通道共用）
        mask = self.bernoulli.sample(x.shape[:1] + (1,) + x.shape[2:]).to(x.device)
        mask = mask / (1 - self.p)  # 缩放补偿

        # 应用到四元数的四个分量 [B,C,H,W] -> 视为 [B,4,D,H,W]
        b, c, h, w = x.shape
        x = x.view(b, 4, c // 4, h, w) * mask.unsqueeze(1)
        return x.view(b, c, h, w)


class HeadTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return torch.tanh(self.conv(x))  # (-1, 1)


class QuaternionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(QuaternionConv2d, self).__init__()
        self.in_channels = in_channels // 4  # 因为每个四元数有四个部分
        self.out_channels = out_channels // 4
        self.padding = padding

        # 初始化四元数卷积的四个分量的权重
        self.r_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.i_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.j_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.k_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))

        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        # 使用适当的初始化策略初始化四元数权重
        nn.init.kaiming_uniform_(self.r_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.i_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.j_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))

    def forward(self, input):
        # 假设输入是形状为 (batch, in_channels, H, W) 的四元数数据
        batch_size, _, height, width = input.size()

        # 拆分输入为四元数组件
        x_r, x_i, x_j, x_k = torch.chunk(input, 4, dim=1)

        # 四元数卷积计算
        conv_rr = F.conv2d(x_r, self.r_weight, stride=1, padding=self.padding)
        conv_ri = F.conv2d(x_r, self.i_weight, stride=1, padding=self.padding)
        conv_rj = F.conv2d(x_r, self.j_weight, stride=1, padding=self.padding)
        conv_rk = F.conv2d(x_r, self.k_weight, stride=1, padding=self.padding)

        conv_ir = F.conv2d(x_i, self.r_weight, stride=1, padding=self.padding)
        conv_ii = F.conv2d(x_i, self.i_weight, stride=1, padding=self.padding)
        conv_ij = F.conv2d(x_i, self.j_weight, stride=1, padding=self.padding)
        conv_ik = F.conv2d(x_i, self.k_weight, stride=1, padding=self.padding)

        conv_jr = F.conv2d(x_j, self.r_weight, stride=1, padding=self.padding)
        conv_ji = F.conv2d(x_j, self.i_weight, stride=1, padding=self.padding)
        conv_jj = F.conv2d(x_j, self.j_weight, stride=1, padding=self.padding)
        conv_jk = F.conv2d(x_j, self.k_weight, stride=1, padding=self.padding)

        conv_kr = F.conv2d(x_k, self.r_weight, stride=1, padding=self.padding)
        conv_ki = F.conv2d(x_k, self.i_weight, stride=1, padding=self.padding)
        conv_kj = F.conv2d(x_k, self.j_weight, stride=1, padding=self.padding)
        conv_kk = F.conv2d(x_k, self.k_weight, stride=1, padding=self.padding)

        # 根据四元数乘法规则合并结果
        out_r = conv_rr - conv_ri - conv_rj - conv_rk
        out_i = conv_ir + conv_ii + conv_ij - conv_ik
        out_j = conv_jr - conv_ji + conv_jj + conv_jk
        out_k = conv_kr + conv_ki - conv_kj + conv_kk

        # 组合输出
        out = torch.cat([out_r, out_i, out_j, out_k], dim=1)
        return out


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


class QuaternionConv2d2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(QuaternionConv2d2, self).__init__()
        self.in_channels = in_channels // 4  # 因为每个四元数有四个部分
        self.out_channels = out_channels // 4

        # 初始化四元数卷积的四个分量的权重
        self.r_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.i_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.j_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.k_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))

        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        # 使用适当的初始化策略初始化四元数权重
        nn.init.kaiming_uniform_(self.r_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.i_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.j_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))

    def forward(self, input):
        # 假设输入是形状为 (batch, in_channels, H, W) 的四元数数据
        batch_size, _, height, width = input.size()

        # 拆分输入为四元数组件
        x_r, x_i, x_j, x_k = torch.chunk(input, 4, dim=1)

        # 四元数卷积计算
        conv_rr = F.conv2d(x_r, self.r_weight, stride=1, padding=0)
        conv_ri = F.conv2d(x_r, self.i_weight, stride=1, padding=0)
        conv_rj = F.conv2d(x_r, self.j_weight, stride=1, padding=0)
        conv_rk = F.conv2d(x_r, self.k_weight, stride=1, padding=0)

        conv_ir = F.conv2d(x_i, self.r_weight, stride=1, padding=0)
        conv_ii = F.conv2d(x_i, self.i_weight, stride=1, padding=0)
        conv_ij = F.conv2d(x_i, self.j_weight, stride=1, padding=0)
        conv_ik = F.conv2d(x_i, self.k_weight, stride=1, padding=0)

        conv_jr = F.conv2d(x_j, self.r_weight, stride=1, padding=0)
        conv_ji = F.conv2d(x_j, self.i_weight, stride=1, padding=0)
        conv_jj = F.conv2d(x_j, self.j_weight, stride=1, padding=0)
        conv_jk = F.conv2d(x_j, self.k_weight, stride=1, padding=0)

        conv_kr = F.conv2d(x_k, self.r_weight, stride=1, padding=0)
        conv_ki = F.conv2d(x_k, self.i_weight, stride=1, padding=0)
        conv_kj = F.conv2d(x_k, self.j_weight, stride=1, padding=0)
        conv_kk = F.conv2d(x_k, self.k_weight, stride=1, padding=0)

        # 根据四元数乘法规则合并结果
        out_r = conv_rr - conv_ri - conv_rj - conv_rk
        out_i = conv_ir + conv_ii + conv_ij - conv_ik
        out_j = conv_jr - conv_ji + conv_jj + conv_jk
        out_k = conv_kr + conv_ki - conv_kj + conv_kk

        # 组合输出
        out = torch.cat([out_r, out_i, out_j, out_k], dim=1)
        return out


def to_image(random_img_out, LH2, HL2, HH2):
    # 初始化逆小波变换对象
    ifm = DWTInverse(wave='bior4.4', mode='zero').to('cuda')

    # 将 LH2, HL2, HH2 组合成一个列表，表示高频子带
    high_freq = torch.stack([LH2, HL2, HH2], dim=2)  # 形状为 [B, C, 3, H, W]

    # 执行逆小波变换
    reconstructed_image = ifm((random_img_out[:, 1:, :, :], [high_freq]))
    return reconstructed_image


class EnhancedQuaternionConv2d(QuaternionConv2d):
    """增强型四元数卷积，强制每个输出通道融合所有输入分量"""

    def __init__(self, in_channels, out_channels, **kwargs):
        # 确保输出通道是4的整数倍
        assert out_channels % 4 == 0, "Output channels must be multiple of 4"
        super().__init__(in_channels, out_channels // 4, **kwargs)
        self.out_channels = out_channels

    def forward(self, x):
        # 标准四元数卷积输出 [N, C*4, H, W]
        base_out = super().forward(x)

        # 增加跨分量融合层
        batch, _, h, w = base_out.shape
        # 重塑为四元数结构 [N, C, 4, H, W]
        quat_view = base_out.view(batch, -1, 4, h, w)

        # 跨分量注意力融合 (Channel-wise Cross-component Attention)
        attn = torch.einsum('ncihw,ncjhw->nij', quat_view, quat_view)  # [N,4,4]
        attn = F.softmax(attn, dim=-1)
        fused = torch.einsum('ncihw,nij->ncjhw', quat_view, attn)  # [N,C,4,H,W]

        # 通道扩展
        expanded = fused.repeat_interleave(4, dim=1)  # [N,C*4,4,H,W]
        return expanded.view(batch, self.out_channels, h, w)


class QuaternionSELayer(nn.Module):
    def __init__(self, channel, reduction=16, cond_dim=0):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            QuaternionConv1d(channel + (cond_dim if cond_dim else 0),
                             channel // reduction, kernel_size=1),
            nn.LeakyReLU(0.2),  # 改用LeakyReLU保留负信息
            QuaternionConv1d(channel // reduction, channel, kernel_size=1),
            nn.LeakyReLU(0.2),  # 改用LeakyReLU保留负信息
        )

    def forward(self, x, cond=None):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        if cond is not None:
            b1, c1, _, _ = cond.size()
            cond = self.avg_pool(cond).view(b1, c1, 1)
            y = torch.cat([y, cond], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HeadLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadLeakyRelu2d, self).__init__()
        self.conv = EnhancedQuaternionConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = EnhancedQuaternionConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = QuaternionBatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.conv2 = EnhancedQuaternionConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = QuaternionBatchNorm2d(out_channels)
        self.se_layer = QuaternionSELayer(out_channels, reduction=8, cond_dim=4)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se_layer(out)
        out += residual
        out = self.activation(out)
        return out


class QuaternionBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(QuaternionBatchNorm2d, self).__init__()
        self.num_features = num_features // 4  # 四元数的通道数是输入的1/4
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            # 可学习的缩放参数（gamma）和偏移参数（beta）
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        # 注册用于运行时的均值和方差的缓冲区
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 检查输入通道数是否是4的倍数（四元数要求）
        if channels % 4 != 0:
            raise ValueError("Input channels must be divisible by 4 for quaternion batch normalization.")

        # 将四元数拆分为实部和虚部
        x_real = x[:, :self.num_features, :, :]  # 实部
        x_imag = x[:, self.num_features:, :, :]  # 虚部

        # 计算实部和虚部的均值和方差
        if self.training:
            mean_real = x_real.mean([0, 2, 3], keepdim=True)
            mean_imag = x_imag.mean([0, 2, 3], keepdim=True)
            var_real = x_real.var([0, 2, 3], keepdim=True, unbiased=False)
            var_imag = x_imag.var([0, 2, 3], keepdim=True, unbiased=False)

            # 更新运行时的均值和方差
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * torch.cat(
                [mean_real.squeeze(), mean_imag.squeeze()], dim=0)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * torch.cat(
                [var_real.squeeze(), var_imag.squeeze()], dim=0)
        else:
            mean_real, mean_imag = torch.split(self.running_mean, self.num_features, dim=0)
            var_real, var_imag = torch.split(self.running_var, self.num_features, dim=0)
            mean_real = mean_real.view(1, -1, 1, 1)
            mean_imag = mean_imag.view(1, -1, 1, 1)
            var_real = var_real.view(1, -1, 1, 1)
            var_imag = var_imag.view(1, -1, 1, 1)

        # 归一化实部和虚部
        x_real_norm = (x_real - mean_real) / torch.sqrt(var_real + self.eps)
        x_imag_norm = (x_imag - mean_imag) / torch.sqrt(var_imag + self.eps)

        # 合并归一化后的实部和虚部
        x_norm = torch.cat([x_real_norm, x_imag_norm], dim=1)

        # 应用可学习的缩放和偏移
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta

        return x_norm


class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        self.encoder = nn.Sequential(
            QuaternionConv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            QuaternionConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            QuaternionConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            QuaternionConv2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 输出像素值在 [0, 1] 范围内
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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
        for i in range(0, len(self.feat_scales)):
            dim = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales) - 1:
                dim_out = get_in_channels([self.feat_scales[i + 1]], inner_channel, channel_multiplier)
                self.decoder.append(
                    AttentionBlock(dim, dim_out)
                )
        # Final head
        self.rgb_decode2 = HeadLeakyRelu2d(64, 64)
        self.rgb_decode1 = HeadTanh2d(64, 9)

    def forward(self, feats, h_in):
        C_LH, C_HL, C_HH = 3, 3, 3
        LH_split, HL_split, HH_split = torch.split(h_in, [C_LH, C_HL, C_HH], dim=1)
        LH_split = to_quaternion(LH_split)
        HL_split = to_quaternion(HL_split)
        HH_split = to_quaternion(HH_split)

        h_in = torch.cat([LH_split, HL_split, HH_split], dim=1)
        # Decoder
        lvl = 0
        for layer in self.decoder:
            if isinstance(layer, Block):
                f_s = feats[0][self.feat_scales[lvl]]  # feature stacked
                if len(self.time_steps) > 1:
                    for i in range(1, len(self.time_steps)):
                        f_s = torch.cat((f_s, feats[i][self.feat_scales[lvl]]), dim=1)
                    f_s = layer(f_s, h_in)
                if lvl != 0:
                    f_s = f_s + x
                lvl += 1
            else:
                f_s = layer(f_s)
                size = feats[0][self.feat_scales[lvl]].size()
                x = F.interpolate(f_s, size=(size[2], size[3]), mode="bilinear", align_corners=True)

        # Fusion Head
        x = self.rgb_decode2(x)
        rgb_img = self.rgb_decode1(x)

        return rgb_img
