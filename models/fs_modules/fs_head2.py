import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sr3_modules.se import ChannelSpatialSELayer
import math
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Conv1d, Sigmoid
from pytorch_wavelets import DWTForward, DWTInverse


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


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=2, dropout_prob=0.1):
        """
        Improved Attention Block with several enhancements:
        - Channel expansion for richer features
        - Dropout for regularization
        - Better residual connection handling
        - Optional skip connection when channel dimensions change

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            expansion_ratio: Ratio for channel expansion in intermediate layers
            dropout_prob: Dropout probability
        """
        super(AttentionBlock, self).__init__()
        self.expanded_channels = out_channels * expansion_ratio

        # First convolution with channel expansion
        self.conv1 = nn.Conv2d(in_channels, self.expanded_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(self.expanded_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(self.expanded_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # Squeeze-and-Excitation layer
        self.se_layer = QuaternionSELayer(out_channels, reduction=8, cond_dim=9)

        # Final convolution
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(out_channels)

        # Activation and dropout
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Skip connection if channel dimensions change
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = None

    def forward(self, x, cond=None):
        # Store original input for residual connection
        identity = x

        # First expansion block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Main processing block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        # Attention mechanism
        out = self.se_layer(out, cond)

        # Handle residual connection
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        out += identity

        # Final processing
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.activation(out)

        return out


class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        # 使用组归一化代替批归一化
        self.norm = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)

        # 主分支使用空洞卷积扩大感受野
        self.block1 = nn.Sequential(
            nn.Conv2d(dim * len(time_steps), dim, kernel_size=3,
                      padding=2, dilation=2),  # 空洞卷积
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.LeakyReLU(0.2)
        )

        # 增强的注意力机制
        self.attention = QuaternionSELayer(dim, reduction=8, cond_dim=9)

        # 使用像素注意力增强细节
        self.pixel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 残差连接使用1x1卷积匹配维度
        self.res_conv = nn.Conv2d(dim * len(time_steps), dim_out, kernel_size=1)

    def forward(self, x, cond=None):
        residual = self.res_conv(x)

        # 主分支
        x = self.block1(x)
        x = self.norm(x)

        # 通道注意力
        x = self.attention(x, cond)

        # 像素注意力
        pa = self.pixel_attention(x)
        x = x * pa

        return F.leaky_relu(x + residual, 0.2)


class HeadTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return torch.tanh(self.conv(x))  # (-1, 1)


def to_image(LL2, LH2, HL2, HH2):
    # 初始化逆小波变换对象
    ifm = DWTInverse(wave='bior4.4', mode='symmetric').to('cuda')

    # 将 LH2, HL2, HH2 组合成一个列表，表示高频子带
    high_freq = torch.stack([LH2, HL2, HH2], dim=2)  # 形状为 [B, C, 3, H, W]

    # 执行逆小波变换
    reconstructed_image = ifm((LL2, [high_freq]))
    return reconstructed_image


class QuaternionSELayer(nn.Module):
    def __init__(self, channel, reduction=16, cond_dim=0):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.max_pool = AdaptiveMaxPool2d(1)  # 新增最大池化分支
        self.fc = Sequential(
            nn.Conv1d(2 * channel + (cond_dim if cond_dim else 0),  # 合并两个池化分支
                      channel // reduction, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()  # 使用Sigmoid确保注意力权重在0-1之间
        )

    def forward(self, x, cond=None):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c, 1)
        max_y = self.max_pool(x).view(b, c, 1)  # 新增最大池化分支
        y = torch.cat([avg_y, max_y], dim=1)  # 合并两个分支

        if cond is not None:
            b1, c1, _, _ = cond.size()
            cond = self.avg_pool(cond).view(b1, c1, 1)
            y = torch.cat([y, cond], dim=1)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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

class HeadLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

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
                Block(dim=dim, dim_out=dim, time_steps=[1])
            )

            if i != len(self.feat_scales) - 1:
                dim_out = get_in_channels([self.feat_scales[i + 1]], inner_channel, channel_multiplier)
                self.decoder.append(
                    AttentionBlock(dim, dim_out)
                )
                self.decoder2.append(
                    AttentionBlock(dim_out, dim_out)
                )
        # Final head
        self.rgb_decode2 = HeadLeakyRelu2d(64, 64)
        self.rgb_decode1 = HeadTanh2d(64, 9)
        self.d = AttentionBlock(64, 64)

    def forward(self, feats, h_in, img):
        C_LH, C_HL, C_HH = 3, 3, 3
        LH_split, HL_split, HH_split = torch.split(h_in, [C_LH, C_HL, C_HH], dim=1)

        h_in = torch.cat([LH_split, HL_split, HH_split], dim=1)
        # Decoder
        lvl = 0
        for layer, layer2 in zip(self.decoder, self.decoder2):
            if isinstance(layer, Block):
                f_s = feats[0][self.feat_scales[lvl]]  # feature stacked
                if len(self.time_steps) > 1:
                    for i in range(1, len(self.time_steps)):
                        f_s = torch.cat((f_s, feats[i][self.feat_scales[lvl]]), dim=1)
                    f_s = layer(f_s, h_in)
                if lvl != 0:
                    f_s = f_s + x
                lvl += 1
                f_s = layer2(f_s, h_in)
            else:
                f_s = layer(f_s, h_in)
                size = feats[0][self.feat_scales[lvl]].size()
                x = F.interpolate(f_s, size=(size[2], size[3]), mode="bilinear", align_corners=True)
                f_s = layer2(f_s, h_in)

        # Fusion Head
        x = self.d(x, h_in)

        x = self.rgb_decode2(x)
        rgb_img = self.rgb_decode1(x)
        C_LH, C_HL, C_HH = 3, 3, 3
        LH_split, HL_split, HH_split = torch.split(rgb_img, [C_LH, C_HL, C_HH], dim=1)
        img = to_image(img, LH_split, HL_split, HH_split)
        return rgb_img, img
