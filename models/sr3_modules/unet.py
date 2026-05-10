import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

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

class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(QuaternionLinear, self).__init__()
        self.in_features = in_features // 4  # 四元数的四个分量 (r, i, j, k)
        self.out_features = out_features // 4

        # 初始化四元数权重 (4, out_features, in_features)
        self.weight = nn.Parameter(
            torch.Tensor(4, self.out_features, self.in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # 拆分输入为四元数组件 (..., in_features) -> 4 x (..., in_features//4)
        x_r, x_i, x_j, x_k = torch.chunk(x, 4, dim=-1)

        # 拆分权重 (4, out_features, in_features)
        w_r, w_i, w_j, w_k = self.weight

        # 构造四元数乘法对应的实矩阵
        # 四元数乘法: (w_r + w_i*i + w_j*j + w_k*k) * (x_r + x_i*i + x_j*j + x_k*k)
        # 结果实部分量: w_r*x_r - w_i*x_i - w_j*x_j - w_k*x_k
        # 结果i分量:    w_r*x_i + w_i*x_r + w_j*x_k - w_k*x_j
        # 结果j分量:    w_r*x_j - w_i*x_k + w_j*x_r + w_k*x_i
        # 结果k分量:    w_r*x_k + w_i*x_j - w_j*x_i + w_k*x_r

        # 构造拼接后的权重矩阵 (4*out_features, 4*in_features)
        cat_weight_r = torch.cat([w_r, -w_i, -w_j, -w_k], dim=-1)  # 实部权重
        cat_weight_i = torch.cat([w_i, w_r, w_k, -w_j], dim=-1)  # i分量权重
        cat_weight_j = torch.cat([w_j, -w_k, w_r, w_i], dim=-1)  # j分量权重
        cat_weight_k = torch.cat([w_k, w_j, -w_i, w_r], dim=-1)  # k分量权重

        # 最终拼接 (4*out_features, 4*in_features)
        cat_weight = torch.cat([cat_weight_r, cat_weight_i, cat_weight_j, cat_weight_k], dim=0)

        # 拼接输入 (..., 4*in_features)
        cat_input = torch.cat([x_r, x_i, x_j, x_k], dim=-1)

        # 线性变换
        out = F.linear(cat_input, cat_weight, self.bias)

        return out

    def extra_repr(self):
        return f'in_features={self.in_features * 4}, out_features={self.out_features * 4}, bias={self.bias is not None}'

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            QuaternionLinear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            QuaternionConv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = QuaternionConv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if (self.with_attn):
            x = self.attn(x)
        return x


class CustomUpsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, target_size):
        # 使用双线性插值或其他方式将x上采样至目标尺寸
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return self.conv(x)


def Reverse(lst):
    return [ele for ele in reversed(lst)]


class UNet(nn.Module):
    def __init__(
            self,
            image_size=128,
            in_channel=18,
            out_channel=9,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                QuaternionLinear(inner_channel, inner_channel * 4),
                Swish(),
                QuaternionLinear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size

        self.init_conv = QuaternionConv2d(in_channels=in_channel, out_channels=inner_channel, kernel_size=3, padding=1)
        downs = []
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, feat_need=False):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        # 记录原始输入尺寸
        original_size = (x.size(2), x.size(3))
        # First downsampling layer
        x = self.init_conv(x)

        # Diffusion encoder
        feats = [x]
        down_sizes = [original_size]  # 记录尺寸
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
                down_sizes.append((x.size(2), x.size(3)))  # 记录每一层后的尺寸
            feats.append(x)


        if feat_need:
            fe = feats.copy()

        # Passing through middle layer
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        # Diffiusion decoder
        if feat_need:
            fd = []
        down_sizes.pop()  # 移除最后一个元素，因为mid不会改变尺寸
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = torch.cat((x, feats.pop()), dim=1)
                x = layer(x, t)
                if feat_need:
                    fd.append(x)
            else:
                target_size = down_sizes.pop()
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)


        # Final Diffusion layer
        x = self.final_conv(x)

        # Output encoder and decoder features if feat_need
        if feat_need:
            return fe, Reverse(fd)
        else:
            return x
