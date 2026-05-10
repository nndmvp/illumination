import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import math
import kornia.color as KC
from matplotlib.colors import Normalize
import numpy as np
import os
from PIL import Image



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


class LocalStatsLoss(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(LocalStatsLoss, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def local_mean_variance(self, x):
        # 计算局部均值
        mean = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

        return mean

    def forward(self, generated, target):
        generated_mean = self.local_mean_variance(generated)
        target_mean = self.local_mean_variance(target)

        mean_loss = F.l1_loss(generated_mean, target_mean)

        return mean_loss


class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cuda')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output



class IlluminationConsistencyLoss(nn.Module):
    def __init__(self):
        super(IlluminationConsistencyLoss, self).__init__()

    def forward(self, image_ir, generate_img):
        """
        Args:
            image_ir: Input noisy image tensor with shape (B, C, W, H).
            generate_img: Generated image tensor with shape (B, C, W, H).
        Returns:
            The illumination consistency loss.
        """
        # Convert images to grayscale to extract illumination information
        image_ir = (image_ir + 1.0) / 2.0
        generate_img = (generate_img + 1.0) / 2.0
        image_ir_gray = self.rgb_to_grayscale(image_ir)
        generate_img_gray = self.rgb_to_grayscale(generate_img)

        # Optionally apply Gaussian blur to focus on low-frequency components
        # image_ir_gray_blur = self.gaussian_blur(image_ir_gray)
        # generate_img_gray_blur = self.gaussian_blur(generate_img_gray)
        # 定义归一化对象，指定原始数据的最小值和最大值
        vmin = image_ir_gray.min()
        vmax = image_ir_gray.max()
        norm_image_ir = (image_ir_gray - vmin) / (vmax - vmin)

        vmin2 = generate_img_gray.min()
        vmax2 = generate_img_gray.max()
        norm_generate_img = (generate_img_gray - vmin2) / (vmax2 - vmin2)

        # Compute the Mean Squared Error (MSE) between the normalized grayscale images
        loss = F.mse_loss(norm_image_ir, norm_generate_img)

        return loss

    @staticmethod
    def rgb_to_grayscale(images):
        """
        Convert RGB images to grayscale.
        Args:
            images: Tensor of shape (B, C, W, H).
        Returns:
            Grayscale images of shape (B, 1, W, H).
        """
        gray_images = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        return gray_images

    @staticmethod
    def gaussian_blur(images, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian blur to the images.
        Args:
            images: Tensor of shape (B, 1, W, H).
            kernel_size: Size of the Gaussian kernel.
            sigma: Standard deviation of the Gaussian kernel.
        Returns:
            Blurred images of shape (B, 1, W, H).
        """
        kernel = IlluminationConsistencyLoss.get_gaussian_kernel(kernel_size, sigma).to(images.device)
        blurred_images = F.conv2d(images, kernel, padding=kernel_size // 2)
        return blurred_images

    @staticmethod
    def get_gaussian_kernel(kernel_size, sigma):
        """
        Generate a 2D Gaussian kernel.
        Args:
            kernel_size: Size of the kernel.
            sigma: Standard deviation of the Gaussian distribution.
        Returns:
            2D Gaussian kernel of shape (1, 1, kernel_size, kernel_size).
        """
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2
        variance = sigma ** 2
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        return gaussian_kernel




class ColourDistanceLoss(nn.Module):
    def __init__(self):
        super(ColourDistanceLoss, self).__init__()

    def forward(self, target_img, generated_img):
        """
        Args:
            target_img: Target image tensor with shape (B, C, W, H).
            generated_img: Generated image tensor with shape (B, C, W, H).
        Returns:
            The colour distance loss.
        """
        # Ensure the input images are in the range [0, 1]
        target_img = (target_img + 1.0) / 2.0
        generated_img = (generated_img + 1.0) / 2.0

        # Extract RGB channels
        R_1, G_1, B_1 = target_img.chunk(3, dim=1)
        R_2, G_2, B_2 = generated_img.chunk(3, dim=1)

        # Calculate the mean of R values
        rmean = (R_1 + R_2) / 2

        # Calculate the differences
        R = R_1 - R_2
        G = G_1 - G_2
        B = B_1 - B_2

        # Calculate the colour distance
        distance = torch.sqrt(
            (2 + rmean / 256) * (R ** 2) +
            4 * (G ** 2) +
            (2 + (255 - rmean) / 256) * (B ** 2)
        )

        # Compute the mean loss
        loss = torch.mean(distance)

        return loss


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.Get_gradient = Get_gradient()
        self.mse_criterion = torch.nn.MSELoss()
        self.mean = LocalStatsLoss()
        self.Get_gradient_nopadding = Get_gradient_nopadding()
        self.cs = ColourDistanceLoss()
        self.illuminationConsistencyLoss = IlluminationConsistencyLoss()

    def forward(self, image_vis, image_ir, generate_img, grad_af, so_img):
        B, C, W, H = image_vis.shape
        image_ir = image_ir.expand(B, C, W, H)

        pred_grad = self.Get_gradient(generate_img)
        pred_x_grad = self.Get_gradient(image_vis)
        loss_grad = self.mse_criterion(pred_grad, pred_x_grad)

        loss_grad_vis = self.Get_gradient_nopadding(image_vis)
        loss_grad_af = self.mse_criterion(loss_grad_vis, grad_af)

        ill_loss = self.illuminationConsistencyLoss(image_vis, generate_img)
        # 损失为颜色差异的平方和的平均值
        # pixel_loss = self.cs(image_ir, generate_img)
        pixel_loss = F.l1_loss(generate_img, so_img)
        return ill_loss * 0, loss_grad*0.01, loss_grad_af * 0, pixel_loss, loss_grad * 0
