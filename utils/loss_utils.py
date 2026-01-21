#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return ((network_output - gt) ** 2).mean()

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Variable:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True
) -> torch.Tensor:
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: Variable,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ... (现有代码保持不变)

# 【新增】双边平滑损失函数
def bilateral_smoothness_loss(roughness: torch.Tensor, image: torch.Tensor, lambda_edge: float = 10.0) -> torch.Tensor:
    """
    roughness: [1, H, W] 预测的粗糙度图
    image: [3, H, W] 引导图像 (通常是 GT Image 或 Albedo)，用于检测边缘
    lambda_edge: 控制边缘敏感度，值越大，边缘处的平滑权重越小 (越允许突变)
    """
    # 1. 计算 Roughness 的梯度 (H 和 W 方向)
    # [1, H, W-1]
    d_rough_x = torch.abs(roughness[:, :, :-1] - roughness[:, :, 1:])
    # [1, H-1, W]
    d_rough_y = torch.abs(roughness[:, :-1, :] - roughness[:, 1:, :])

    # 2. 计算引导 Image 的梯度 (取 RGB 平均变成灰度，或者保持 3 通道取最大值)
    # 这里简单取 RGB 平均作为亮度/结构引导
    img_gray = image.mean(dim=0, keepdim=True)  # [1, H, W]

    d_img_x = torch.abs(img_gray[:, :, :-1] - img_gray[:, :, 1:])  # [1, H, W-1]
    d_img_y = torch.abs(img_gray[:, :-1, :] - img_gray[:, 1:, :])  # [1, H-1, W]

    # 3. 计算双边权重
    # 图像梯度越大 (边缘)，exp(-x) 越接近 0 -> 权重越小 -> 允许 Roughness 不平滑
    # 图像梯度越小 (平坦)，exp(-x) 越接近 1 -> 权重越大 -> 强迫 Roughness 平滑
    weights_x = torch.exp(-lambda_edge * d_img_x)
    weights_y = torch.exp(-lambda_edge * d_img_y)

    # 4. 加权求和
    loss_x = (weights_x * d_rough_x).mean()
    loss_y = (weights_y * d_rough_y).mean()

    return loss_x + loss_y