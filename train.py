import os
import csv
import json
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Dict, List, Optional, Tuple, Union
import math
import time

import kornia
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm, trange
from diff_gaussian_rasterization import Gaussian_SSR
import gc
from lpips import LPIPS

from arguments import GroupParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene import GaussianModel, Scene, Camera
from utils.general_utils import safe_state,save_pbr_debug_montage
from utils.image_utils import psnr, turbo_cmap, erode
from utils.loss_utils import l1_loss, ssim, get_img_grad_weight, bilateral_smoothness_loss, hsv_albedo_loss
from utils.graphics_utils import normal_from_depth_image

from utils.warp_utils import warp_consistency_loss,material_consistency_loss
import random

import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def align_disparity_simple(disp_source, disp_target, mask):
    """
    输入两个都是视差图 (近大远小)，计算 scale/offset 使 source 拟合 target
    """
    # 筛选有效点
    source_masked = disp_source[mask]
    target_masked = disp_target[mask]

    # 计算中位数
    t_source = torch.median(source_masked)
    t_target = torch.median(target_masked)

    # 计算平均绝对偏差 (MAD)
    s_source = torch.mean(torch.abs(source_masked - t_source))
    s_target = torch.mean(torch.abs(target_masked - t_target))

    # 计算参数
    eps = 1e-8
    if s_source < eps:
        scale = 1.0
        offset = 0.0
    else:
        scale = s_target / s_source
        offset = t_target - scale * t_source

    # 应用对齐
    aligned_source = scale * disp_source + offset

    # 截断负值 (视差必须 >= 0)
    aligned_source = torch.clamp(aligned_source, min=0.0)

    return aligned_source, scale, offset

def align_depth_robust(mono_disp, render_depth, mask):
    """
    输入:
    mono_disp: 单目先验 (Depth Anything 输出, 本质是视差, 近大远小)
    render_depth: 3DGS 渲染深度 (物理深度, 近小远大)
    mask: 有效区域 [1, H, W]

    输出:
    aligned_mono_disp: 对齐到 3DGS 尺度的单目视差
    """
    # 1. 维度防御 (确保都是 [1, H, W])
    if render_depth.ndim == 2: render_depth = render_depth.unsqueeze(0)
    if mono_disp.ndim == 2: mono_disp = mono_disp.unsqueeze(0)
    if mask.ndim == 2: mask = mask.unsqueeze(0)

    # 2. 数据准备：统一转到【视差空间】
    eps = 1e-6

    # Depth Anything 输出本身就是视差，不需要取倒数 (除非你之前手动取过倒数，否则直接用)
    # 假设输入是 [0, 1] 范围的视差
    disp_mono = mono_disp

    # 3DGS 渲染的是物理深度，必须取倒数变成视差
    disp_render = 1.0 / (render_depth + eps)

    # 3. 筛选有效点计算 Scale/Offset
    valid_mask = mask & (disp_mono > eps) & (render_depth > eps) & \
                 torch.isfinite(disp_mono) & torch.isfinite(disp_render)

    if valid_mask.sum() < 10:
        return disp_mono, 1.0, 0.0

    dm_masked = disp_mono[valid_mask]
    dr_masked = disp_render[valid_mask]

    # 4. 计算统计量 (Median & MAD)
    t_mono = torch.median(dm_masked)
    s_mono = torch.mean(torch.abs(dm_masked - t_mono))
    t_render = torch.median(dr_masked)
    s_render = torch.mean(torch.abs(dr_masked - t_render))

    # 5. 计算对齐参数
    # 目标: aligned_mono ≈ disp_render
    if s_mono < 1e-7:
        scale = 1.0
        offset = 0.0
    else:
        scale = s_render / s_mono
        offset = t_render - scale * t_mono

    # 6. 应用对齐
    aligned_mono_disp = scale * disp_mono + offset

    # 截断负值，视差不能为负
    aligned_mono_disp = torch.clamp(aligned_mono_disp, min=eps)

    return aligned_mono_disp, scale, offset

def pearson_correlation_loss(pred, target, mask=None):
    """
    计算皮尔逊相关系数 Loss。
    Loss = 1 - Correlation
    范围 [0, 2]，0 表示完全正相关（形状一致），1 表示无关，2 表示负相关。
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    # 避免空数据或数据太少
    if pred.numel() < 10:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # 1. 减去均值 (Center the data) -> 消除 Shift 影响
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # 2. 计算标准差 (Std) -> 消除 Scale 影响
    # 加上 1e-8 防止除以零
    pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)

    # 3. 计算相关系数
    correlation = (pred_centered * target_centered).sum() / (pred_std * target_std + 1e-8)

    # 4. 返回 Loss (我们要最大化相关性，即最小化 1 - r)
    return 1.0 - correlation


def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale],
                                            intrinsic_matrix.to(depth.device),
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref


def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        # srgb1 = 1.055 * torch.pow(torch.clamp(linear, min=eps), 1.0/2.4) - 0.055
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

def srgb_to_linear(srgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(srgb, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        linear0 = 25 / 323 * srgb
        linear1 = ((srgb + 0.055) / 1.055)**2.4
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        linear0 = 25 / 323 * srgb
        linear1 = ((srgb + 0.055) / 1.055)**2.4
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError

def get_tv_loss(
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    pad: int = 1,
    step: int = 1,
) -> torch.Tensor:
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(
                -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            rgb_grad_w = torch.exp(
                -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss


def get_masked_tv_loss(
    mask: torch.Tensor,  # [1, H, W]
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    erosion: bool = False,
) -> torch.Tensor:
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]

    # erode mask
    mask = mask.float()
    if erosion:
        kernel = mask.new_ones([7, 7])
        mask = kornia.morphology.erosion(mask[None, ...], kernel)[0]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]

    tv_loss = (tv_h * rgb_grad_h * mask_h).mean() + (tv_w * rgb_grad_w * mask_w).mean()

    return tv_loss


#用于生成环境贴图（Environment Map）的球面方向向量矩阵，主要用于将二维环境贴图转换为三维方向向量，支持基于物理的渲染计算
def get_envmap_dirs(res: List[int] = [512, 1024]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
    return reflvec


def resize_tensorboard_img(
    img: torch.Tensor,  # [C, H, W]
    max_res: int = 800,
) -> torch.Tensor:
    _, H, W = img.shape
    ratio = min(max_res / H, max_res / W)
    target_size = (int(H * ratio), int(W * ratio))
    transform = T.Resize(size=target_size)
    img = transform(img)  # [C, H', W']
    return img


def depth_gradient(depth):
    # depth shape: [1, H, W]
    # 计算 x 方向梯度: d(x+1) - d(x)
    dy = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])
    dx = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])

    # 填充回原尺寸 (为了保持形状一致，简单 padding)
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))  # Pad H
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))  # Pad W
    return dy, dx

#动态权重调度
def get_prior_weight(current_iter, start_iter, decay_start_iter, end_iter, base_weight, min_ratio=0.01):
    """
    计算当前步数的权重：
    1. iter < start: 0 (预热)
    2. start <= iter < decay_start: base_weight (全速)
    3. decay_start <= iter < end: 线性衰减到 base_weight * min_ratio
    4. iter >= end: base_weight * min_ratio (保持微弱约束)
    """
    if current_iter < start_iter:
        return 0.0
    if current_iter < decay_start_iter:
        return base_weight
    if current_iter >= end_iter:
        return base_weight * min_ratio

    # 计算衰减进度 (0.0 -> 1.0)
    progress = (current_iter - decay_start_iter) / (end_iter - decay_start_iter)
    # 线性衰减
    # current_ratio = 1.0 - (1.0 - min_ratio) * progress

    # 或者使用 Cosine 衰减 (更平滑，推荐)
    import math
    cosine_progress = (math.cos(progress * math.pi) + 1.0) * 0.5  # 1 -> 0
    current_ratio = min_ratio + (1.0 - min_ratio) * cosine_progress

    return base_weight * current_ratio




def get_normal_propagation_bounds(args: Namespace, pbr_iteration: int) -> Tuple[int, int]:
    """Return the absolute iteration range for reflection-driven normal propagation."""
    prop_start = getattr(args, "normal_prop_start", 0)
    if prop_start <= 0:
        prop_start = pbr_iteration + 1
    prop_iters = max(0, getattr(args, "normal_prop_iters", 0))
    return prop_start, prop_start + prop_iters


def is_normal_propagation_active(args: Namespace, iteration: int, pbr_iteration: int) -> bool:
    if not getattr(args, "use_normal_propagation", False):
        return False
    if iteration <= pbr_iteration:
        return False
    prop_start, prop_end = get_normal_propagation_bounds(args, pbr_iteration)
    return iteration >= prop_start and iteration < prop_end


def set_gaussian_param_lr(gaussians: GaussianModel, group_names, lr: float) -> None:
    """Set optimizer LR for selected Gaussian parameter groups if they exist."""
    if gaussians.optimizer is None:
        return
    if isinstance(group_names, str):
        group_names = {group_names}
    else:
        group_names = set(group_names)
    for param_group in gaussians.optimizer.param_groups:
        if param_group.get("name") in group_names:
            param_group["lr"] = lr


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def build_normal_propagation_mask(
    roughness_map: torch.Tensor,
    specular_map: torch.Tensor,
    normal_mask: torch.Tensor,
    args: Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build a conservative reliability mask for reflection-driven normal propagation."""
    with torch.no_grad():
        low_rough_mask = roughness_map.detach() < args.normal_prop_rough_thresh
        spec_intensity = specular_map.detach().max(dim=0, keepdim=True)[0]
        high_spec_mask = spec_intensity > args.normal_prop_spec_thresh
        valid_normal_mask = normal_mask.detach().bool()

        if args.normal_prop_mask_mode == "rough_or_spec":
            prop_mask = (low_rough_mask | high_spec_mask) & valid_normal_mask
        elif args.normal_prop_mask_mode == "rough_only":
            prop_mask = low_rough_mask & valid_normal_mask
        elif args.normal_prop_mask_mode == "spec_only":
            prop_mask = high_spec_mask & valid_normal_mask
        else:
            prop_mask = low_rough_mask & high_spec_mask & valid_normal_mask

        image_ratio = prop_mask.float().mean().item()
        valid_pixels = valid_normal_mask.float().sum().clamp_min(1.0)
        valid_ratio = (prop_mask.float().sum() / valid_pixels).item()

        if valid_ratio < args.normal_prop_min_valid_ratio or valid_ratio > args.normal_prop_max_valid_ratio:
            prop_mask = torch.zeros_like(prop_mask, dtype=torch.bool)

        stats = {
            "image_ratio": image_ratio,
            "valid_ratio": valid_ratio,
            "rough_ratio": (low_rough_mask & valid_normal_mask).float().sum().div(valid_pixels).item(),
            "spec_ratio": (high_spec_mask & valid_normal_mask).float().sum().div(valid_pixels).item(),
            "skipped": float(prop_mask.sum().item() == 0),
        }
    return prop_mask, stats


def save_normal_prop_debug(
    save_dir: str,
    iteration: int,
    gt_image: torch.Tensor,
    render_rgb: torch.Tensor,
    normal_map: torch.Tensor,
    normal_map_from_depth: torch.Tensor,
    roughness_map: torch.Tensor,
    specular_map: torch.Tensor,
    prop_mask: torch.Tensor,
    save_individual: bool = False,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        gt = gt_image.detach().cpu().clamp(0, 1)
        render = render_rgb.detach().cpu().clamp(0, 1)
        error = torch.abs(render - gt)
        error = error / (error.max() + 1e-6)
        normal = ((normal_map.detach().cpu() + 1.0) * 0.5).clamp(0, 1)
        depth_normal = ((normal_map_from_depth.detach().cpu() + 1.0) * 0.5).clamp(0, 1)
        rough = roughness_map.detach().cpu().clamp(0, 1).repeat(3, 1, 1)
        spec = specular_map.detach().cpu().max(dim=0, keepdim=True)[0].clamp(0, 1).repeat(3, 1, 1)
        mask = prop_mask.detach().cpu().float().repeat(3, 1, 1)
        mask_overlay = (0.65 * render + 0.35 * torch.tensor([1.0, 0.0, 0.0])[:, None, None] * mask).clamp(0, 1)
        row1 = torch.cat([gt, render, error, mask_overlay], dim=2)
        row2 = torch.cat([normal, depth_normal, rough, spec], dim=2)
        torchvision.utils.save_image(torch.cat([row1, row2], dim=1), os.path.join(save_dir, f"iter_{iteration:05d}_normal_prop.jpg"))

        if save_individual:
            iter_dir = os.path.join(save_dir, f"iter_{iteration:05d}")
            os.makedirs(iter_dir, exist_ok=True)
            images = {
                "gt": gt,
                "render": render,
                "error": error,
                "mask_overlay": mask_overlay,
                "normal": normal,
                "normal_from_depth": depth_normal,
                "roughness": rough,
                "specular": spec,
                "prop_mask": mask,
            }
            for name, image_tensor in images.items():
                torchvision.utils.save_image(image_tensor, os.path.join(iter_dir, f"{name}.png"))


class NormalPropagationDebugLogger:
    """Write file-based normal-propagation diagnostics for non-TensorBoard workflows."""

    def __init__(self, model_path: str, args: Namespace):
        self.enabled = getattr(args, "use_normal_propagation", False) and getattr(args, "normal_prop_log_interval", 0) > 0
        self.log_interval = max(1, getattr(args, "normal_prop_log_interval", 1))
        self.drop_threshold = getattr(args, "normal_prop_advice_drop_psnr", 0.5)
        self.reference_psnr_ema = None
        self.active_psnr_ema = None
        self.ema_decay = 0.95
        self.csv_path = None
        self.jsonl_path = None
        self.summary_path = None

        if self.enabled:
            self.debug_dir = os.path.join(model_path, "debug_normal_propagation")
            os.makedirs(self.debug_dir, exist_ok=True)
            self.csv_path = os.path.join(self.debug_dir, "metrics.csv")
            self.jsonl_path = os.path.join(self.debug_dir, "diagnosis.jsonl")
            self.summary_path = os.path.join(self.debug_dir, "latest_summary.json")
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self._fieldnames())
                    writer.writeheader()

    @staticmethod
    def _fieldnames() -> List[str]:
        return [
            "iteration",
            "active",
            "train_psnr",
            "loss",
            "pbr_l1",
            "valid_ratio",
            "image_ratio",
            "rough_ratio",
            "spec_ratio",
            "skipped",
            "grad_raw_norm",
            "grad_masked_norm",
            "grad_keep_ratio",
            "normal_lr",
            "material_lr",
            "light_lr",
            "recommendation",
        ]

    def _update_ema(self, value: float, active: bool) -> None:
        if active:
            current = self.active_psnr_ema
            self.active_psnr_ema = value if current is None else self.ema_decay * current + (1.0 - self.ema_decay) * value
        else:
            current = self.reference_psnr_ema
            self.reference_psnr_ema = value if current is None else self.ema_decay * current + (1.0 - self.ema_decay) * value

    def _recommend(self, active: bool, stats: Dict[str, float], grad_stats: Dict[str, float]) -> str:
        if not active:
            return "collecting_reference"
        if stats.get("skipped", 0.0) > 0.5:
            if stats.get("valid_ratio", 0.0) < 1e-6:
                return "mask_empty: lower --normal_prop_spec_thresh or try --normal_prop_mask_mode rough_or_spec"
            return "mask_ratio_guard_triggered: inspect thresholds or widen min/max valid-ratio bounds"
        if self.reference_psnr_ema is not None and self.active_psnr_ema is not None:
            drop = self.reference_psnr_ema - self.active_psnr_ema
            if drop > self.drop_threshold:
                return "psnr_drop: reduce --normal_prop_normal_grad_scale, increase --normal_prop_ramp_iters, or use spec_only"
        grad_keep = grad_stats.get("masked", 0.0) / max(grad_stats.get("raw", 0.0), 1e-12)
        if grad_keep > 0.8 and stats.get("valid_ratio", 0.0) > 0.1:
            return "broad_gradient: reduce --normal_prop_max_valid_ratio or switch to rough_and_spec/spec_only"
        if grad_keep < 1e-4 and stats.get("valid_ratio", 0.0) > 0.0:
            return "weak_gradient: inspect PBR/specular signal or raise --normal_prop_normal_grad_scale carefully"
        return "ok"

    def record(
        self,
        iteration: int,
        active: bool,
        train_psnr: float,
        loss_value: float,
        pbr_l1: Optional[float],
        stats: Dict[str, float],
        grad_stats: Dict[str, float],
        lr_stats: Dict[str, float],
    ) -> None:
        if not self.enabled:
            return

        self._update_ema(train_psnr, active)
        recommendation = self._recommend(active, stats, grad_stats)
        should_log = iteration % self.log_interval == 0 or recommendation not in {"ok", "collecting_reference"}
        if not should_log:
            return

        grad_keep_ratio = grad_stats.get("masked", 0.0) / max(grad_stats.get("raw", 0.0), 1e-12)
        row = {
            "iteration": iteration,
            "active": int(active),
            "train_psnr": train_psnr,
            "loss": loss_value,
            "pbr_l1": "" if pbr_l1 is None else pbr_l1,
            "valid_ratio": stats.get("valid_ratio", 0.0),
            "image_ratio": stats.get("image_ratio", 0.0),
            "rough_ratio": stats.get("rough_ratio", 0.0),
            "spec_ratio": stats.get("spec_ratio", 0.0),
            "skipped": stats.get("skipped", 1.0),
            "grad_raw_norm": grad_stats.get("raw", 0.0),
            "grad_masked_norm": grad_stats.get("masked", 0.0),
            "grad_keep_ratio": grad_keep_ratio,
            "normal_lr": lr_stats.get("normal", 0.0),
            "material_lr": lr_stats.get("material", 0.0),
            "light_lr": lr_stats.get("light", 0.0),
            "recommendation": recommendation,
        }

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames())
            writer.writerow(row)

        summary = {
            **row,
            "reference_psnr_ema": self.reference_psnr_ema,
            "active_psnr_ema": self.active_psnr_ema,
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        if recommendation not in {"ok", "collecting_reference"}:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(summary) + "\n")
            print(f"[Normal Propagation Debug] Iter {iteration}: {recommendation}")


def get_optimizer_group_lr(optimizer: Optional[torch.optim.Optimizer], group_name: str) -> float:
    if optimizer is None:
        return 0.0
    for param_group in optimizer.param_groups:
        if param_group.get("name") == group_name:
            return float(param_group.get("lr", 0.0))
    return 0.0


def get_optimizer_first_lr(optimizer: Optional[torch.optim.Optimizer]) -> float:
    if optimizer is None or len(optimizer.param_groups) == 0:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


class MultiViewConsistencyDebugLogger:
    """File logger for multi-view material consistency without relying on TensorBoard."""

    def __init__(self, model_path: str, args: Namespace):
        self.enabled = getattr(args, "use_consistency", False) and getattr(args, "consistency_log_interval", 0) > 0
        self.log_interval = max(1, getattr(args, "consistency_log_interval", 1))
        self.csv_path = None
        self.summary_path = None
        if self.enabled:
            self.debug_dir = os.path.join(model_path, "debug_consistency")
            os.makedirs(self.debug_dir, exist_ok=True)
            self.csv_path = os.path.join(self.debug_dir, "metrics.csv")
            self.summary_path = os.path.join(self.debug_dir, "latest_summary.json")
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self._fieldnames())
                    writer.writeheader()

    @staticmethod
    def _fieldnames() -> List[str]:
        return [
            "iteration", "src_uid", "tgt_uid", "rank", "loss", "weighted_loss",
            "valid_ratio", "valid_pixels", "grid_ratio", "occ_ratio", "edge_ratio",
            "facing_ratio", "src_depth_ratio", "tgt_depth_ratio", "rough_l1",
            "metal_l1", "albedo_l1", "skipped", "recommendation",
        ]

    @staticmethod
    def recommend(stats: Dict[str, float], min_valid_ratio: float, max_valid_ratio: float) -> str:
        valid_ratio = stats.get("valid_ratio", 0.0)
        if stats.get("skipped", 0.0) > 0.5 or valid_ratio < min_valid_ratio:
            return "too_few_matches: use closer views, relax depth/edge thresholds, or lower --consistency_min_valid_ratio"
        if valid_ratio > max_valid_ratio:
            return "mask_too_broad: tighten occlusion/edge thresholds or lower --consistency_max_valid_ratio"
        if stats.get("occ_ratio", 0.0) < 0.2:
            return "occlusion_filter_strict_or_bad_projection: inspect debug masks and camera/depth scale"
        if stats.get("edge_ratio", 0.0) < 0.2:
            return "edge_filter_strict: increase --consistency_edge_rel_thresh if debug edges look valid"
        if stats.get("rough_l1", 0.0) + stats.get("metal_l1", 0.0) > 0.5:
            return "large_material_disagreement: lower --lambda_consistency or delay --consistency_start"
        return "ok"

    def record(
        self,
        iteration: int,
        src_uid: int,
        tgt_uid: int,
        rank: int,
        loss_value: float,
        weighted_loss: float,
        stats: Dict[str, float],
        min_valid_ratio: float,
        max_valid_ratio: float,
    ) -> None:
        if not self.enabled:
            return
        recommendation = self.recommend(stats, min_valid_ratio, max_valid_ratio)
        if iteration % self.log_interval != 0 and recommendation == "ok":
            return
        row = {
            "iteration": iteration,
            "src_uid": src_uid,
            "tgt_uid": tgt_uid,
            "rank": rank,
            "loss": loss_value,
            "weighted_loss": weighted_loss,
            "valid_ratio": stats.get("valid_ratio", 0.0),
            "valid_pixels": stats.get("valid_pixels", 0.0),
            "grid_ratio": stats.get("grid_ratio", 0.0),
            "occ_ratio": stats.get("occ_ratio", 0.0),
            "edge_ratio": stats.get("edge_ratio", 0.0),
            "facing_ratio": stats.get("facing_ratio", 0.0),
            "src_depth_ratio": stats.get("src_depth_ratio", 0.0),
            "tgt_depth_ratio": stats.get("tgt_depth_ratio", 0.0),
            "rough_l1": stats.get("rough_l1", 0.0),
            "metal_l1": stats.get("metal_l1", 0.0),
            "albedo_l1": stats.get("albedo_l1", 0.0),
            "skipped": stats.get("skipped", 1.0),
            "recommendation": recommendation,
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames())
            writer.writerow(row)
        with open(self.summary_path, "w") as f:
            json.dump(row, f, indent=2)
        if recommendation != "ok":
            print(f"[Consistency Debug] Iter {iteration}: {recommendation}")


def build_camera_uid_to_index(train_cameras: List[Camera]) -> Dict[int, int]:
    return {int(cam.uid): idx for idx, cam in enumerate(train_cameras)}


def select_consistency_target(
    viewpoint_cam: Camera,
    train_cameras: List[Camera],
    knn_map,
    uid_to_index: Dict[int, int],
    rank_min: int,
    rank_max: int,
) -> Tuple[Camera, int]:
    src_idx = uid_to_index.get(int(viewpoint_cam.uid), None)
    if src_idx is not None and len(train_cameras) > 1:
        actual_min = max(1, min(rank_min, len(train_cameras) - 1))
        actual_max = max(actual_min, min(rank_max, len(train_cameras) - 1))
        rank = random.randint(actual_min, actual_max)
        tgt_idx = int(knn_map[src_idx][rank])
        return train_cameras[tgt_idx], rank

    # Fallback for datasets whose uid no longer matches the train camera list.
    candidates = [cam for cam in train_cameras if cam is not viewpoint_cam]
    if not candidates:
        return viewpoint_cam, 0
    return random.choice(candidates), -1


def training(
    dataset: GroupParams,
    opt: GroupParams,
    pipe: GroupParams,
    testing_iterations: List[int],
    saving_iterations: List[int],
    checkpoint_iterations: int,
    checkpoint_path: Optional[str] = None,
    pbr_iteration: int = 7_000,
    debug_from: int = -1,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
    normal_tv_weight: float = 1.0,
    brdf_tv_weight: float = 1.0,
    env_tv_weight: float = 0.01,
    radius: float = 0.8,
    bias: float = 0.01,
    thick: float = 0.05,
    delta: float = 0.0625,
    step: int = 16,
    start: int = 8,
    indirect: bool = False,
# 【新增】接收位置约束参数
    use_position_opt: bool = False,
) -> None:
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    tb_writer = prepare_output_and_logger(dataset)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # NOTE: prepare for PBR
    brdf_lut = get_brdf_lut().cuda()
    envmap_dirs = get_envmap_dirs()
    cubemap = CubemapLight(base_res=256).cuda()
    cubemap.train()
    # print(torch.isnan(cubemap.base).any())

    param_groups = [
        {"name": "cubemap", "params": cubemap.parameters(), "lr": opt.opacity_lr}
    ]
    light_optimizer = torch.optim.Adam(param_groups, lr=opt.opacity_lr)

    canonical_rays = scene.get_canonical_rays()

    # load checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model_params = checkpoint["gaussians"]
        first_iter = checkpoint["iteration"]
        # cubemap_params = checkpoint["cubemap"]
        # light_optimizer_params = checkpoint["light_optimizer"]
        # irradiance_volumes_params = checkpoint["irradiance_volumes"]

        gaussians.restore(model_params, opt)
        # cubemap.load_state_dict(cubemap_params)
        # light_optimizer.load_state_dict(light_optimizer_params)
        print(f"Load checkpoint from {checkpoint_path}")

    # define progress bar
    viewpoint_stack = None
    progress_bar = trange(first_iter, opt.iterations, desc="Training progress")  # For logging
    use_mono_depth = getattr(dataset, "use_mono_depth", False)
    use_mono_normal = getattr(dataset, "use_mono_normal", False)

    print("Start training...单目深度权重为",getattr(opt, "lambda_mono", 0.1) , "单目法线权重为",getattr(opt, "lambda_mono_normal", 0.1))
    print("深度先验：",use_mono_depth,"法线先验：",use_mono_normal)

    # 训练循环外的初始化
    max_specular_count = 0
    sabotage_patience_counter = 0
    sabotage_active = args.color_sabotage

    # --- 配置参数 (可以提取到 args 里) ---
    # 深度先验调度
    depth_start = 1000
    depth_decay_start = 7000
    depth_end = 25000
    depth_min_ratio = 0.05  # 深度最后保留 5% 的力度，防止几何彻底崩坏

    # 法线先验调度
    normal_start = 1000
    normal_decay_start = 5000  # 法线衰减要早一点
    normal_end = 15000  # 法线结束要早一点
    normal_min_ratio = 0.0  # 法线最后建议完全关闭，否则会影响高频纹理

    normal_prop_debug_logger = NormalPropagationDebugLogger(args.model_path, args)
    consistency_debug_logger = MultiViewConsistencyDebugLogger(args.model_path, args)
    if getattr(args, "use_normal_propagation", False):
        prop_start, prop_end = get_normal_propagation_bounds(args, pbr_iteration)
        print(
            f"[Normal Propagation] enabled from iter {prop_start} to {prop_end - 1}; "
            f"normal_lr={args.normal_prop_normal_lr}, "
            f"material_lr_scale={args.normal_prop_material_lr_scale}, "
            f"light_lr_scale={args.normal_prop_light_lr_scale}, "
            f"rough_thresh={args.normal_prop_rough_thresh}, "
            f"spec_thresh={args.normal_prop_spec_thresh}, "
            f"mask_mode={args.normal_prop_mask_mode}, "
            f"valid_ratio=[{args.normal_prop_min_valid_ratio}, {args.normal_prop_max_valid_ratio}], "
            f"ramp_iters={args.normal_prop_ramp_iters}, "
            f"log_interval={args.normal_prop_log_interval}"
        )

    # =========================================================
    # [Pre-compute] 基于 3D 空间距离的 KNN 邻居表
    # =========================================================
    print("正在计算相机空间距离矩阵 (Spatial KNN)...")

    train_cameras = scene.getTrainCameras()
    num_cams = len(train_cameras)

    # 1. 提取所有相机中心 [N, 3]
    # 注意：确保 train_cameras 里的顺序在之后不会变
    cam_centers = torch.stack([cam.camera_center for cam in train_cameras]).to("cuda")

    # 2. 计算距离矩阵 [N, N]
    # dists[i, j] 代表第 i 个相机和第 j 个相机的距离
    dists = torch.cdist(cam_centers[None, ...], cam_centers[None, ...]).squeeze(0)

    # 3. 获取排序后的索引 [N, N]
    # sorted_indices[i, 0] 是自己 (距离为0)
    # sorted_indices[i, 1] 是最近的邻居
    # sorted_indices[i, 2] 是第二近的邻居...
    _, sorted_indices = torch.sort(dists, dim=1)

    # 转回 CPU 存起来，省显存
    # 这里存的是 train_cameras 列表里的下标索引
    knn_map = sorted_indices.cpu().numpy()
    camera_uid_to_index = build_camera_uid_to_index(train_cameras)

    print("空间邻居表构建完成。")
    # =========================================================


    for iteration in range(first_iter + 1, opt.iterations + 1):  # the real iteration (1 shift)
        iter_start.record()
        normal_prop_active = is_normal_propagation_active(args, iteration, pbr_iteration)
        if getattr(args, "use_normal_propagation", False):
            if normal_prop_active:
                set_gaussian_param_lr(gaussians, "normal", args.normal_prop_normal_lr)
                material_lr = opt.BRDF_lr * args.normal_prop_material_lr_scale
                set_gaussian_param_lr(gaussians, ["albedo", "roughness", "metallic"], material_lr)
                set_optimizer_lr(light_optimizer, opt.opacity_lr * args.normal_prop_light_lr_scale)
            else:
                set_gaussian_param_lr(gaussians, "normal", opt.opacity_lr)
                if iteration > pbr_iteration:
                    set_gaussian_param_lr(gaussians, ["albedo", "roughness", "metallic"], opt.BRDF_lr)
                set_optimizer_lr(light_optimizer, opt.opacity_lr)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        try:
            c2w = torch.inverse(viewpoint_cam.world_view_transform.T)  # [4, 4]
        except:
            continue

        # Render
        # if (iteration - 1) == debug_from:
        #     pipe.debug = True
        pipe.debug

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration <= pbr_iteration:
            background = bg
        else:  # NOTE: black background for PBR
            background = torch.zeros_like(bg)
        rendering_result = render(
            viewpoint_camera=viewpoint_cam,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            pad_normal=False,
            derive_normal=True,
            radius=radius,
            bias=bias,
            thick=thick,
            delta=delta,
            step=step,
            start=start
        )

        tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)
        image_height=int(viewpoint_cam.image_height)
        image_width=int(viewpoint_cam.image_width)

        image = rendering_result["render"]  # [3, H, W]
        viewspace_point_tensor = rendering_result["viewspace_points"]
        visibility_filter = rendering_result["visibility_filter"]
        radii = rendering_result["radii"]
        normal_map_from_depth = rendering_result["normal_map_from_depth"]  # [3, H, W]  根据深度图导出的法线图
        normal_map = rendering_result["normal_map"]  # [3, H, W] 法线属性渲染出的G-Buffer
        albedo_map = rendering_result["albedo_map"]  # [3, H, W]
        roughness_map = rendering_result["roughness_map"]  # [1, H, W]
        metallic_map = rendering_result["metallic_map"]  # [1, H, W]
        # allmap = rendering_result["allmap"]
        #specular_map = rendering_result["metallic_map"]

        # formulate roughness
        rmax, rmin = 1.0, 0.04
        roughness_map = roughness_map * (rmax - rmin) + rmin

        # NOTE: mask normal map by view direction to avoid skip value
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]

        # Loss

        alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image[0:3, :, :].cuda()
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

        loss: torch.Tensor
        Ll1 = F.l1_loss(image, gt_image)
        normal_loss = 0.0
        loss_mono_depth = 0.0
        loss_omnidata = 0.0
        loss_consist = 0.0  # 【新增】初始化一致性 Loss
        consistency_stats = {
            "valid_ratio": 0.0,
            "valid_pixels": 0.0,
            "grid_ratio": 0.0,
            "occ_ratio": 0.0,
            "edge_ratio": 0.0,
            "facing_ratio": 0.0,
            "src_depth_ratio": 0.0,
            "tgt_depth_ratio": 0.0,
            "rough_l1": 0.0,
            "metal_l1": 0.0,
            "albedo_l1": 0.0,
            "skipped": 1.0,
        }
        consistency_tgt_uid = -1
        consistency_rank = -1

        aligned_mono_depth = None
        train_render_for_metrics = image
        pbr_render_loss_for_debug = None
        normal_prop_stats = {
            "image_ratio": 0.0,
            "valid_ratio": 0.0,
            "rough_ratio": 0.0,
            "spec_ratio": 0.0,
            "skipped": 1.0,
        }
        normal_prop_grad_stats = {"raw": 0.0, "masked": 0.0}



        # --- 计算当前权重 ---
        # 这里的 lambda_mono 和 lambda_mono_normal 是你在命令行设置的基础权重(如 0.1, 0.05)
        cur_lambda_depth = get_prior_weight(
            iteration, depth_start, depth_decay_start, depth_end,
            args.lambda_mono, depth_min_ratio
        )

        cur_lambda_normal = get_prior_weight(
            iteration, normal_start, normal_decay_start, normal_end,
            args.lambda_mono_normal, normal_min_ratio
        )

        if iteration <= pbr_iteration:
            # --- 基础 RGB & Normal Loss ---
            mask = rendering_result["normal_from_depth_mask"]
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            # === 原有的normal_loss保持不变 ===
            normal_loss_weight = 1.0
            normal_loss = F.l1_loss(normal_map[:, mask], normal_map_from_depth[:, mask])
            loss += normal_loss_weight * normal_loss
            normal_tv_loss = get_tv_loss(gt_image, normal_map, pad=1, step=1)
            loss += normal_tv_loss * normal_tv_weight

            if iteration % 1000 == 0:
                print(f"[Scheduler] Iter {iteration}: Depth W={cur_lambda_depth:.4f}, Normal W={cur_lambda_normal:.4f}")

            # >>>>> 深度监督模块 (Depth Anything) <<<<<
            # 前 1000 步 Warm-up 跳过，防止几何不稳定导致 Scale 计算错误
            if use_mono_depth and hasattr(viewpoint_cam, 'mono_depth_image'):

                # 1. 数据准备 & 维度标准化 [1, H, W]
                # -----------------------------------------------------------
                mono_disp = viewpoint_cam.mono_depth_image  # 原始输入本质是视差
                if mono_disp.ndim == 2:
                    mono_disp = mono_disp.unsqueeze(0)
                elif mono_disp.shape[0] == 3:
                    mono_disp = mono_disp[0:1, ...]
                elif mono_disp.shape[2] == 3:
                    mono_disp = mono_disp[..., 0:1].permute(2, 0, 1)

                render_depth = rendering_result["depth_map"]  # 物理深度 (近小远大)
                render_disp = 1.0 / (render_depth + 1e-6)  # 转换为视差 (近大远小) 用于对齐和Loss

                # 2. 生成 Valid Mask
                # -----------------------------------------------------------
                # A. 基础几何 Mask
                gt_alpha = viewpoint_cam.gt_alpha_mask.cuda()
                if gt_alpha.ndim == 2: gt_alpha = gt_alpha.unsqueeze(0)
                render_mask = mask if mask.ndim == 3 else mask.unsqueeze(0)

                valid_mask = (mono_disp > 1e-4) & render_mask & (gt_alpha > 0.5)

                # B. RGB 亮度剔除 (去除过曝/欠曝区域的错误估计)
                if hasattr(viewpoint_cam, 'original_image'):
                    gt_image_raw = viewpoint_cam.original_image.cuda()
                    intensity = gt_image_raw.mean(dim=0, keepdim=True)
                    valid_mask &= (intensity > 0.02) & (intensity < 0.99)

                # 3. 对齐与 Loss 计算
                # -----------------------------------------------------------
                if valid_mask.sum() > 100:
                    # A. 视差对齐 (在视差空间进行线性回归)
                    # aligned_mono_depth 在这里实际上是 aligned_mono_disp
                    aligned_mono_disp, scale, offset = align_disparity_simple(
                        mono_disp, render_disp.detach(), valid_mask
                    )
                    # 赋值给外部变量用于可能的 Debug 可视化
                    aligned_mono_depth = aligned_mono_disp

                    # B. 计算边缘感知权重 (降低边缘处的 Loss 权重)
                    dy, dx = depth_gradient(aligned_mono_disp)
                    grad_mag = torch.sqrt(dy ** 2 + dx ** 2)
                    grad_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
                    edge_weight = torch.exp(-5.0 * grad_norm)

                    # C. 计算深度 Loss
                    depth_loss_type = getattr(opt, "depth_loss_type", "complex")

                    if depth_loss_type == 'l1':
                        # 简单 L1
                        diff = torch.abs(render_disp - aligned_mono_disp)
                        loss_depth_final = (diff * edge_weight)[valid_mask].mean()
                    else:
                        # 组合 Loss: Log L1 + Pearson + Gradient
                        # (1) Log L1 Loss
                        diff_map = torch.abs(render_disp - aligned_mono_disp)
                        log_loss = (torch.log(1.0 + diff_map) * edge_weight)[valid_mask].mean()

                        # (2) Pearson Correlation Loss (结构一致性)
                        pred_v = render_disp[valid_mask]
                        target_v = aligned_mono_disp[valid_mask]
                        pred_v = pred_v - pred_v.mean()
                        target_v = target_v - target_v.mean()
                        # 余弦相似度
                        loss_pearson = 1.0 - (
                                (pred_v * target_v).sum() /
                                (torch.sqrt((pred_v ** 2).sum() * (target_v ** 2).sum()) + 1e-8)
                        )

                        # (3) Gradient Loss (仅后期启用，增强细节)
                        loss_grad = 0.0
                        if iteration > 3000:
                            g_render_y, g_render_x = depth_gradient(render_disp)
                            g_mono_y, g_mono_x = depth_gradient(aligned_mono_disp)
                            g_diff = torch.abs(g_render_y - g_mono_y) + torch.abs(g_render_x - g_mono_x)
                            loss_grad = g_diff[valid_mask].mean()

                        # 组合权重: L1主导，Pearson保结构，Grad保锐度
                        loss_depth_final = 0.7 * log_loss + 0.3 * loss_pearson + 0.05 * loss_grad

                    # 4. 异常剔除与累加
                    # -----------------------------------------------------------
                    loss_mono_depth = loss_depth_final

                    # 帧级异常检测 (阈值可根据 log loss 的量级调整，通常 1.0-2.0)
                    if loss_mono_depth.item() > 1.5:
                        if iteration % 100 == 0:
                            print(f"[Warn] Iter {iteration} Depth Loss {loss_mono_depth.item():.4f} > 1.5. Ignored.")
                        loss_mono_depth = 0.0
                    else:
                        loss += cur_lambda_depth * loss_mono_depth

                    # ==================== 论文专用插图生成模块 ====================
                    # 每 500 次迭代输出一次论文用图（可自行修改频率，比如 500）
                    if iteration % 500 == 0 and valid_mask.sum() > 100:
                        paper_dir = os.path.join(args.model_path, "paper_figures")
                        os.makedirs(paper_dir, exist_ok=True)

                        # -------------------------------------------------
                        # 论文图 3-1：深度尺度对齐前后对比图 (散点图)
                        # -------------------------------------------------
                        try:
                            import matplotlib
                            matplotlib.use('Agg')  # 强制使用无头后端，彻底绕过 Linux 报错
                            import matplotlib.pyplot as plt
                            import matplotlib.font_manager as fm
                            import scipy.stats as stats

                            # 挂载我们刚刚测试成功的黑体
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            simhei_path = os.path.join(current_dir, 'simsun.ttc')

                            if not os.path.exists(simhei_path):
                                print(f"[Warn] 找不到 {simhei_path}，中文可能显示异常！")
                                zh_font = fm.FontProperties(size=12)
                                zh_font_title = fm.FontProperties(size=14)
                            else:
                                zh_font = fm.FontProperties(fname=simhei_path, size=12)
                                zh_font_title = fm.FontProperties(fname=simhei_path, size=14)

                            plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

                            # 提取掩膜内的数据，转为 Numpy 数组
                            target_vals = render_disp[valid_mask].detach().cpu().numpy()
                            source_vals_unaligned = mono_disp[valid_mask].detach().cpu().numpy()
                            source_vals_aligned = aligned_mono_disp[valid_mask].detach().cpu().numpy()

                            # 随机采样最多 10000 个点避免画图过慢和重叠
                            sample_size = min(10000, len(target_vals))
                            indices = np.random.choice(len(target_vals), sample_size, replace=False)

                            # 计算量化指标 RMSE
                            rmse_unaligned = np.sqrt(np.mean((source_vals_unaligned - target_vals) ** 2))
                            rmse_aligned = np.sqrt(np.mean((source_vals_aligned - target_vals) ** 2))

                            plt.figure(figsize=(18, 5))

                            # 子图 1：对齐前
                            plt.subplot(1, 3, 1)
                            plt.scatter(target_vals[indices], source_vals_unaligned[indices], alpha=0.3, s=2,
                                        c='#1f77b4')
                            ax_min, ax_max = target_vals[indices].min(), target_vals[indices].max()
                            plt.plot([ax_min, ax_max], [ax_min, ax_max], 'r--', label='y=x (理想)')
                            plt.xlabel('3DGS 渲染视差 (目标值)', fontproperties=zh_font)
                            plt.ylabel('预测视差 (未对齐)', fontproperties=zh_font)
                            plt.title(f'对齐前 (RMSE: {rmse_unaligned:.2f})', fontproperties=zh_font_title)
                            plt.legend(prop=zh_font)

                            # 子图 2：对齐后
                            plt.subplot(1, 3, 2)
                            plt.scatter(target_vals[indices], source_vals_aligned[indices], alpha=0.3, s=2, c='#ff7f0e')
                            plt.plot([ax_min, ax_max], [ax_min, ax_max], 'r--', label='y=x (理想)')
                            plt.xlabel('3DGS 渲染视差 (目标值)', fontproperties=zh_font)
                            plt.ylabel('预测视差 (对齐后)', fontproperties=zh_font)
                            plt.title(f'对齐后 (RMSE: {rmse_aligned:.2f})', fontproperties=zh_font_title)
                            plt.legend(prop=zh_font)

                            # 子图 3：误差分布直方图
                            plt.subplot(1, 3, 3)
                            error_unaligned = source_vals_unaligned - target_vals
                            error_aligned = source_vals_aligned - target_vals

                            hist_min = np.percentile(error_unaligned, 5)
                            hist_max = np.percentile(error_unaligned, 95)
                            bins = np.linspace(hist_min, hist_max, 50)

                            plt.hist(error_unaligned, bins=bins, alpha=0.5, label='对齐前误差', color='#1f77b4',
                                     density=True)
                            plt.hist(error_aligned, bins=bins, alpha=0.5, label='对齐后误差', color='#ff7f0e',
                                     density=True)
                            plt.axvline(x=0, color='r', linestyle='--')
                            plt.xlabel('绝对误差 (预测值 - 目标值)', fontproperties=zh_font)
                            plt.ylabel('密度', fontproperties=zh_font)
                            plt.title('对齐前后残差分布对比', fontproperties=zh_font_title)
                            plt.legend(prop=zh_font)

                            plt.tight_layout()
                            plt.savefig(f"{paper_dir}/iter_{iteration:05d}_alignment_analysis.png", dpi=300)
                            plt.close()
                        except ImportError:
                            print("[Warn] matplotlib 未安装，无法生成深度对齐散点图，请 pip install matplotlib")

                        # -------------------------------------------------
                        # 论文图 3-2：预测差异与边缘掩膜生成过程图 (横向拼接)
                        # -------------------------------------------------
                        def to_color(tensor_2d, mask_2d):
                            """将灰度的深度/视差图转为美观的 Turbo 伪彩图"""
                            valid_vals = tensor_2d[mask_2d]
                            if len(valid_vals) == 0:
                                return torch.zeros((3, tensor_2d.shape[0], tensor_2d.shape[1]))
                            # 使用分位数掐头去尾，使色彩对比度最大化
                            vmin = torch.quantile(valid_vals, 0.05).item()
                            vmax = torch.quantile(valid_vals, 0.95).item()
                            norm_tensor = (tensor_2d - vmin) / (vmax - vmin + 1e-6)
                            norm_tensor = torch.clamp(norm_tensor, 0, 1)
                            # turbo_cmap 需要在 CPU 运算
                            color_np = turbo_cmap(norm_tensor.cpu().numpy())  # [H, W, 3]
                            color_th = torch.from_numpy(color_np).permute(2, 0, 1).float()
                            return color_th * mask_2d.cpu().float()

                        mask_2d = valid_mask.squeeze(0).detach()  # [H, W]

                        # 1. 原始 RGB 图
                        rgb_vis = gt_image.detach().cpu()  # [3, H, W]

                        # 2. 单目预测深度 (对齐后，用于清晰展示其平滑但边缘模糊的特点)
                        aligned_disp_vis = to_color(aligned_mono_disp.squeeze(0).detach(), mask_2d)

                        # 3. 3DGS 渲染深度 (包含细碎噪声或不准的区域)
                        render_disp_vis = to_color(render_disp.squeeze(0).detach(), mask_2d)

                        # 4. 生成的边缘掩膜 (Mask / Weight)
                        # 使用了原代码中的 edge_weight [1, H, W]
                        weight_vis = edge_weight.squeeze(0).detach().cpu()  # [H, W]
                        # 转成 3 通道灰度图，并只保留有效掩膜区域
                        weight_vis = weight_vis.unsqueeze(0).repeat(3, 1, 1) * mask_2d.cpu().float()

                        # 横向拼接：[ RGB | 单目伪彩 | 3DGS 渲染伪彩 | 边缘权重掩膜 ]
                        row_img = torch.cat([rgb_vis, aligned_disp_vis, render_disp_vis, weight_vis], dim=2)

                        # 保存过程图
                        torchvision.utils.save_image(row_img, f"{paper_dir}/iter_{iteration:05d}_mask_process.png")
                    # ==================== 论文专用插图生成模块结束 ====================

            # ==================== 深度与视差双重 Debug ====================
            if use_mono_depth and iteration % 500 == 0:
                # 1. 创建两个文件夹
                dir_disp = os.path.join(args.model_path, "debug_disparity")
                dir_depth = os.path.join(args.model_path, "debug_metric_depth")
                os.makedirs(dir_disp, exist_ok=True)
                os.makedirs(dir_depth, exist_ok=True)

                # 2. 数据准备
                # ---------------------------------------------------------
                # A. 原始数据获取
                render_depth_raw = rendering_result["depth_map"].detach()  # [1, H, W] 物理深度
                aligned_disp_raw = aligned_mono_depth.detach() if aligned_mono_depth is not None else None  # [1, H, W] 视差

                if aligned_disp_raw is not None:
                    # B. 维度标准化 (统一为 [1, H, W])
                    if render_depth_raw.ndim == 2: render_depth_raw = render_depth_raw.unsqueeze(0)
                    if aligned_disp_raw.ndim == 2:
                        aligned_disp_raw = aligned_disp_raw.unsqueeze(0)
                    elif aligned_disp_raw.ndim == 3 and aligned_disp_raw.shape[0] == 3:
                        aligned_disp_raw = aligned_disp_raw[0:1]

                    # C. 尺寸对齐 (以 3DGS 渲染分辨率为准)
                    if render_depth_raw.shape[-2:] != aligned_disp_raw.shape[-2:]:
                        aligned_disp_raw = F.interpolate(
                            aligned_disp_raw.unsqueeze(0),
                            size=render_depth_raw.shape[-2:],
                            mode='bilinear', align_corners=False
                        ).squeeze(0)

                    # ---------------------------------------------------------
                    # 3. 核心可视化函数 (通用)
                    # ---------------------------------------------------------
                    def save_visualization(pred, target, mask, save_path, mode='disp'):
                        """
                        mode='disp': 视差模式，关注近处，截断极大值
                        mode='depth': 深度模式，关注远处，截断极大值(远平面)
                        """
                        # 1. 动态计算截断阈值
                        if mask.sum() > 0:
                            clip_max = torch.quantile(target[mask], 0.98).item()
                            min_limit = 1.0 if mode == 'disp' else 10.0
                            clip_max = max(clip_max, min_limit)
                        else:
                            clip_max = 10.0 if mode == 'disp' else 100.0

                        def robust_norm(d):
                            d_clamped = torch.clamp(d, max=clip_max)
                            valid_d = d_clamped[mask]
                            if valid_d.numel() > 0:
                                d_min, d_max = valid_d.min(), valid_d.max()
                                if (d_max - d_min) > 1e-6:
                                    return (d_clamped - d_min) / (d_max - d_min)
                            return torch.zeros_like(d)

                        vis_pred = robust_norm(pred)
                        vis_target = robust_norm(target)

                        # 误差图
                        diff = torch.abs(pred - target)
                        vis_diff = robust_norm(diff)

                        # 【修复点在这里】先乘 Mask，再拼接
                        # 这样每一张小图都单独被 Mask 处理了，背景变黑
                        vis_pred = vis_pred * mask
                        vis_target = vis_target * mask
                        vis_diff = vis_diff * mask

                        # 拼接: [Pred | Target | Diff] -> [1, H, 3W]
                        row = torch.cat([vis_pred, vis_target, vis_diff], dim=2)
                        # 复制成 3 通道 (RGB) 以便保存 -> [3, H, 3W]
                        row = row.repeat(3, 1, 1)
                        torchvision.utils.save_image(row, save_path)

                    # ---------------------------------------------------------
                    # 4. 生成两组可视化
                    # ---------------------------------------------------------

                    # === Group 1: 视差对比 (Disparity) ===
                    # 3DGS 深度转视差
                    pred_disp = 1.0 / (render_depth_raw + 1e-6)
                    target_disp = aligned_disp_raw

                    save_visualization(
                        pred_disp, target_disp, valid_mask,
                        f"{dir_disp}/iter_{iteration:05d}_disp.png",
                        mode='disp'
                    )

                    # === Group 2: 深度对比 (Metric Depth) ===
                    # 单目视差转深度
                    pred_depth = render_depth_raw
                    target_depth = 1.0 / (aligned_disp_raw + 1e-6)

                    save_visualization(
                        pred_depth, target_depth, valid_mask,
                        f"{dir_depth}/iter_{iteration:05d}_depth.png",
                        mode='depth'
                    )

            # [新增] 单目法线监督
            # ================= [Mn: 基于法线一致性的 Mask] =================
            if use_mono_normal and hasattr(viewpoint_cam, 'mono_normal_image'):

                # 1. 获取数据 [3, H, W]
                gt_normal = viewpoint_cam.mono_normal_image  # OmniData 先验
                pred_normal = rendering_result["normal_map"]  # 3DGS 渲染法线

                # 2. 计算法线一致性 (Cosine Similarity) 参考TSGS
                # 点积: 1.0 = 完全一致, 0.0 = 垂直, -1.0 = 反向
                # dot: [1, H, W]
                dot_prod = (pred_normal * gt_normal).sum(dim=0, keepdim=True)

                # 3. 生成 Mn (Consistency Mask)
                # 逻辑：只监督那些方向“大体一致”的区域，剔除完全离谱的 Outliers
                # 这里的阈值 (consistency_thresh) 决定了容忍度
                # 例如 0.5 表示：夹角小于 60 度才算有效，超过 60 度认为是先验错误/遮挡，不给 Loss
                consistency_thresh = 0.5
                Mn_consistency = (dot_prod > consistency_thresh).float()

                # 4. 结合物体 Mask (GT Alpha)
                gt_alpha_mask = (viewpoint_cam.gt_alpha_mask.cuda() > 0.5).float()
                if gt_alpha_mask.ndim == 2: gt_alpha_mask = gt_alpha_mask.unsqueeze(0)

                # 最终 Mask = 法线一致性 * 物体掩码
                Mn = Mn_consistency * gt_alpha_mask

                # ================= [计算 Loss] =================
                if Mn.sum() > 100:
                    # 使用 1-NN Loss
                    # 注意：这里我们用截断后的 dot_prod 来算 Loss
                    dot_clamped = torch.clamp(dot_prod, min=-1.0, max=1.0)
                    loss_cosine = 1.0 - dot_clamped

                    # 只在 Mn 区域内计算
                    # 这样做的效果：只有当 3DGS 的法线和 OmniData 法线“稍微有点像”的时候，
                    # 我们才推它一把让它“更像”。如果完全不像，就放过它（避免被错误先验带偏）。
                    loss_omnidata = (loss_cosine * Mn).sum() / (Mn.sum() + 1e-6)

                    # ================= 单帧异常剔除 =================
                    # 设定阈值。Cosine Loss 范围是 [0, 2]。
                    # 0.6 表示平均误差很大 (约等于平均夹角很大)，通常意味着方向全反了或坐标系不对
                    normal_loss_threshold = 0.6
                    current_normal_loss_val = loss_omnidata.item()
                    if current_normal_loss_val > normal_loss_threshold:
                        if iteration % 100 == 0:
                            print(
                                f"[Warning Iter {iteration}] Normal Loss ({current_normal_loss_val:.4f}) > {normal_loss_threshold}. Ignoring normal prior for this frame.")

                        # 【核心】直接归零
                        loss_omnidata = 0.0
                    else:
                        # 正常情况
                        loss += cur_lambda_normal * loss_omnidata

                    # (可选) Debug
                    # if iteration % 1000 == 0:
                    #     print(f"Normal Consistency Kept: {Mn.sum()/gt_alpha_mask.sum():.2%}")

            # ==================== 法线对比 Debug====================
            if use_mono_normal and iteration % 500 == 0:
                if hasattr(viewpoint_cam, 'mono_normal_image') and viewpoint_cam.mono_normal_image is not None:

                    debug_dir = os.path.join(args.model_path, "debug_normals")
                    os.makedirs(debug_dir, exist_ok=True)

                    with torch.no_grad():
                        # 1. 获取预测法线 & GT 法线
                        pred_normal = rendering_result["normal_map"].detach()  # [3, H, W]
                        gt_normal = viewpoint_cam.mono_normal_image  # [3, H, W]

                        # 2. 获取并处理 Mask (关键步骤)
                        # 复用 Loss 计算时的 Mask 逻辑，确保可视化和 Loss 一致
                        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
                        if gt_alpha_mask.ndim == 2:
                            gt_alpha_mask = gt_alpha_mask.unsqueeze(0)  # [1, H, W]

                        vis_mask = (gt_alpha_mask > 0.5).float()

                        # 3. 维度对齐
                        if pred_normal.shape[-2:] != gt_normal.shape[-2:]:
                            gt_normal = torch.nn.functional.interpolate(gt_normal.unsqueeze(0),
                                                                        size=pred_normal.shape[-2:],
                                                                        mode='bilinear',
                                                                        align_corners=False).squeeze(0)
                            gt_normal = torch.nn.functional.normalize(gt_normal, dim=0)

                            # Mask 也要对齐尺寸
                            vis_mask = torch.nn.functional.interpolate(vis_mask.unsqueeze(0),
                                                                       size=pred_normal.shape[-2:],
                                                                       mode='nearest').squeeze(0)

                        # 4. 可视化转换: [-1, 1] -> [0, 1]
                        vis_pred = (pred_normal + 1.0) * 0.5
                        vis_gt = (gt_normal + 1.0) * 0.5

                        # ========== [核心修改: 应用 Mask] ==========
                        # 将被 Mask 掉的区域变成黑色 (乘以 0)
                        vis_pred_masked = vis_pred * vis_mask
                        vis_gt_masked = vis_gt * vis_mask
                        # =========================================

                        # 5. 计算误差图
                        dot_prod = (pred_normal * gt_normal).sum(dim=0, keepdim=True)
                        error_map = 1.0 - torch.clamp(dot_prod, min=-1.0, max=1.0)

                        # 误差图也应用 Mask，忽略掉无效区域的误差显示
                        error_map = error_map * vis_mask

                        error_map_vis = error_map / (error_map.mean() * 5.0 + 1e-6)
                        error_map_vis = torch.clamp(error_map_vis, 0, 1)
                        vis_error = error_map_vis.repeat(3, 1, 1)

                        #保存mask图像
                        torchvision.utils.save_image(vis_mask.float(),
                                                     f"{debug_dir}/step_{iteration:05d}_mask_check.png")

                        #print(f"[DEBUG] Mask saved. Valid pixels: {vis_mask.sum()} / {vis_mask.numel()}")

                        # 6. 拼接: [渲染(Masked) | GT(Masked) | 误差(Masked)]
                        comparison = torch.cat([vis_pred_masked, vis_gt_masked, vis_error], dim=2)

                        torchvision.utils.save_image(comparison,
                                                     f"{debug_dir}/step_{iteration:05d}_normal_masked.png")

                        # 可选: 如果你想看具体的 Cosine Loss 数值
                        # avg_cos_loss = error_map.mean().item()
                        # print(f"[DEBUG] Saved normal check to {debug_dir}/step_{iteration:05d}_normal.png (Avg Cos Error: {avg_cos_loss:.4f})")
            # ==================== [Debug 代码结束] ====================

        else:  # NOTE: PBR
            # recon occlusion
            if indirect:
                occlusion = rendering_result["occlusion_map"].permute(1, 2, 0)
            else:
                occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]

            out_normal_view = rendering_result["out_normal_view"]
            depth_pos = rendering_result["depth_pos"]
            normal_mask = rendering_result["normal_mask"]  # [1, H, W]
            cubemap.build_mips() # build mip for environment light

            # 3DGS-DR-style normal propagation: only this scheduled window lets
            # reflection/PBR gradients flow back into the learned Gaussian normals.
            normals_for_pbr = normal_map.permute(1, 2, 0)  # [H, W, 3]
            if not normal_prop_active:
                normals_for_pbr = normals_for_pbr.detach()

            pbr_result = pbr_shading(
                light=cubemap,
                normals=normals_for_pbr,
                view_dirs=view_dirs,
                mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
                albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
                roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
                metallic=metallic_map.permute(1, 2, 0) if metallic else None,  # [H, W, 1]
                tone=tone,
                gamma=gamma,
                occlusion=occlusion.detach(),
                brdf_lut=brdf_lut,
            )

            diffuse_rgb = (
                pbr_result["diffuse_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
            )

            diffuse_rgb = torch.where(
                normal_mask,
                diffuse_rgb,
                background[:, None, None]
            )

            render_direct = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
            render_direct = torch.where(
                normal_mask,
                render_direct,
                background[:, None, None],
            )

            specular_map = pbr_result["specular_rgb"].permute(2, 0, 1)

            normal_prop_valid_ratio = 0.0
            normal_prop_stats = {
                "image_ratio": 0.0,
                "valid_ratio": 0.0,
                "rough_ratio": 0.0,
                "spec_ratio": 0.0,
                "skipped": 1.0,
            }
            normal_prop_grad_stats = {"raw": 0.0, "masked": 0.0}
            if normal_prop_active:
                prop_mask, normal_prop_stats = build_normal_propagation_mask(
                    roughness_map=roughness_map,
                    specular_map=specular_map,
                    normal_mask=normal_mask,
                    args=args,
                )
                normal_prop_valid_ratio = normal_prop_stats["valid_ratio"]

                # Mask reflection-driven normal gradients to reliable glossy/specular pixels.
                # A ramp avoids a sudden PSNR drop when PBR normal gradients are first enabled.
                normal_grad_mask = prop_mask.permute(1, 2, 0).float().expand_as(normals_for_pbr)
                prop_start, _ = get_normal_propagation_bounds(args, pbr_iteration)
                ramp = min(1.0, max(0.0, (iteration - prop_start + 1) / max(1, args.normal_prop_ramp_iters)))
                normal_grad_scale = args.normal_prop_normal_grad_scale * ramp

                def normal_prop_hook(grad, mask=normal_grad_mask, scale=normal_grad_scale, stats=normal_prop_grad_stats):
                    masked_grad = grad * mask * scale
                    stats["raw"] = grad.detach().norm().item()
                    stats["masked"] = masked_grad.detach().norm().item()
                    return masked_grad

                normals_for_pbr.register_hook(normal_prop_hook)

                # Low-LR/freeze material decomposition during propagation so the
                # image residual is less likely to be absorbed by BRDF parameters.
                material_grad_scale = args.normal_prop_material_grad_scale
                if material_grad_scale != 1.0:
                    albedo_map.register_hook(lambda grad, scale=material_grad_scale: grad * scale)
                    roughness_map.register_hook(lambda grad, scale=material_grad_scale: grad * scale)
                    metallic_map.register_hook(lambda grad, scale=material_grad_scale: grad * scale)

            # =========================================================
            # [Feature] 自监督高光掩膜 (Self-Supervised Specular Masking)
            # =========================================================
            # 只有在开启参数且 albedo 是可导变量时才执行
            if args.use_specular_mask:  # 建议加个 warmup，前期别开
                # A. 计算高光强度 (取 RGB 亮度的最大值)
                spec_intensity = specular_map.detach().max(dim=0, keepdim=True)[0]  # [1, H, W]
                # B. 制作梯度权重 (Gradient Weight)
                grad_scale = torch.exp(-5.0 * spec_intensity.clamp(0, 1))
                # C. 注册 Hook (梯度拦截器)
                def hook_fn(grad):
                    return grad * grad_scale

                # 注册 Hook
                albedo_map.register_hook(hook_fn)
            # =========================================================

            SSR = Gaussian_SSR(tanfovx, tanfovy, image_width, image_height, radius, bias, thick, delta, step, start)
            if metallic:
                F0 = (1.0 - metallic_map) * 0.04 + albedo_map * metallic_map
            else:
                F0 = torch.ones_like(albedo_map) * 0.04  # [1, H, W, 3]
                metallic_map = torch.zeros_like(roughness_map)

            linear_rgb = srgb_to_linear(render_direct)

            (IRR, _) = SSR(out_normal_view.detach(), depth_pos.detach(), linear_rgb.detach(), albedo_map, roughness_map, metallic_map, F0)
            IRR = linear_to_srgb(IRR)
            IRR = kornia.filters.median_blur(IRR[None, ...], (3, 3))[0]
            render_rgb = render_direct + IRR
            train_render_for_metrics = render_rgb

            if normal_prop_active and args.normal_prop_debug_interval > 0 and iteration % args.normal_prop_debug_interval == 0:
                save_normal_prop_debug(
                    save_dir=os.path.join(args.model_path, "debug_normal_propagation"),
                    iteration=iteration,
                    gt_image=gt_image,
                    render_rgb=render_rgb,
                    normal_map=normal_map,
                    normal_map_from_depth=normal_map_from_depth,
                    roughness_map=roughness_map,
                    specular_map=specular_map,
                    prop_mask=prop_mask,
                    save_individual=args.normal_prop_save_individual_images,
                )

            pbr_render_loss = l1_loss(render_rgb, gt_image)
            pbr_render_loss_for_debug = pbr_render_loss.detach().item()
            loss = pbr_render_loss

            if normal_prop_active and args.normal_prop_depth_weight > 0:
                prop_depth_mask = rendering_result["normal_from_depth_mask"]
                if prop_depth_mask.sum() > 100:
                    loss += args.normal_prop_depth_weight * F.l1_loss(
                        normal_map[:, prop_depth_mask],
                        normal_map_from_depth[:, prop_depth_mask],
                    )

            ### BRDF loss
            if (normal_mask == 0).sum() > 0:
                brdf_tv_loss = get_masked_tv_loss(
                    normal_mask,
                    gt_image,  # [3, H, W]
                    torch.cat([albedo_map, roughness_map, metallic_map], dim=0),  # [5, H, W]
                )
            else:
                brdf_tv_loss = get_tv_loss(
                    gt_image,  # [3, H, W]
                    torch.cat([albedo_map, roughness_map, metallic_map], dim=0),  # [5, H, W]
                    pad=1,  # FIXME: 8 for scene
                    step=1,
                )
            loss += brdf_tv_loss * brdf_tv_weight
            lamb_weight = 0.01
            lamb_loss = (1.0 - roughness_map[normal_mask]).mean() + metallic_map[normal_mask].mean()
            loss += lamb_loss * lamb_weight

            # ==================================================================
            # [Feature] HSV约束
            # ==================================================================
            if args.lambda_hsv > 0 and (normal_mask.sum() > 0):
                # 使用 HSV Loss 约束 Albedo
                # 这里的参数很有讲究:
                # lambda_h=1.0: 强迫色相一致 (颜色不能偏)
                # lambda_s=0.5: 饱和度大概一致就行
                # lambda_v=0.1: 对亮度只做平滑约束 (TV Loss), 绝对不要让它去拟合 GT 的亮度值！
                loss_hsv = hsv_albedo_loss(
                    albedo_map,
                    gt_image,
                    lambda_h=1.0,
                    lambda_s=0.5,
                    lambda_v=0.1
                )
                loss += loss_hsv * args.lambda_hsv


            # ==================================================================
            # [Feature] 全局双边平滑约束 (Global Bilateral Smoothness)
            # 目标: Albedo, Roughness, Metallic, Normal
            # ==================================================================
            if args.use_bilateral_loss and args.lambda_bilateral > 0 and (normal_mask.sum() > 0):
                # 1. 准备引导图 (GT Image 是最稳的 Ground Truth)
                # detach 很重要，防止梯度传回 GT 导致逻辑混乱，虽然 GT 本身也没梯度
                guide_img = gt_image.detach()
                edge_sens = args.bilateral_edge

                # 2. 计算各分量的平滑 Loss
                # Roughness: 强制区域一致性 (Case 1)
                loss_bi_rough = bilateral_smoothness_loss(roughness_map, guide_img, lambda_edge=edge_sens)
                # Metallic: 去除黑白椒盐噪点
                loss_bi_metal = bilateral_smoothness_loss(metallic_map, guide_img, lambda_edge=edge_sens)

                # 3. 汇总求和
                total_smooth_loss = loss_bi_rough + loss_bi_metal
                loss += total_smooth_loss * args.lambda_bilateral

            # ==================================================================
            # [New Feature] 多视角一致性 (Optimized + Debug Version)
            # ==================================================================
            consistency_start = args.consistency_start if args.consistency_start > 0 else pbr_iteration + args.consistency_start_offset
            consistency_active = (
                args.use_consistency
                and args.lambda_consistency > 0
                and iteration >= consistency_start
                and (iteration % max(1, args.consistency_interval) == 0)
            )
            if consistency_active:
                render_pkg_src = rendering_result
                viewpoint_src = viewpoint_cam

                debug_dir = None
                if args.consistency_debug_interval > 0 and iteration % args.consistency_debug_interval == 0:
                    debug_dir = os.path.join(args.model_path, "debug_consistency")
                    print(f"[Info] Debug consistency image will be saved to {debug_dir}")

                viewpoint_tgt, consistency_rank = select_consistency_target(
                    viewpoint_cam=viewpoint_cam,
                    train_cameras=train_cameras,
                    knn_map=knn_map,
                    uid_to_index=camera_uid_to_index,
                    rank_min=args.consistency_rank_min,
                    rank_max=args.consistency_rank_max,
                )
                consistency_tgt_uid = int(getattr(viewpoint_tgt, "uid", -1))

                with torch.no_grad():
                    render_pkg_tgt = render(viewpoint_tgt, gaussians, pipe, background)

                src_maps = {
                    'roughness': render_pkg_src['roughness_map'],
                    'metallic': render_pkg_src['metallic_map'],
                    'albedo': render_pkg_src['albedo_map'],
                }
                tgt_maps = {
                    'roughness': render_pkg_tgt['roughness_map'],
                    'metallic': render_pkg_tgt['metallic_map'],
                    'albedo': render_pkg_tgt['albedo_map'],
                }

                loss_mat_ref_gs, consistency_stats = material_consistency_loss(
                    src_maps, render_pkg_src['depth_map'], viewpoint_src,
                    tgt_maps, render_pkg_tgt['depth_map'], viewpoint_tgt,
                    src_normal=render_pkg_src.get('normal_map', None),
                    src_valid_mask=render_pkg_src.get('normal_mask', None),
                    tgt_valid_mask=render_pkg_tgt.get('normal_mask', None),
                    constraint_albedo=args.consistency_albedo,
                    save_debug_path=debug_dir,
                    iteration=iteration,
                    occlusion_abs_thresh=args.consistency_occ_abs_thresh,
                    occlusion_rel_thresh=args.consistency_occ_rel_thresh,
                    edge_rel_thresh=args.consistency_edge_rel_thresh,
                    facing_thresh=args.consistency_facing_thresh,
                    robust_eps=args.consistency_robust_eps,
                    return_stats=True,
                )

                if (
                    consistency_stats["valid_ratio"] >= args.consistency_min_valid_ratio
                    and consistency_stats["valid_ratio"] <= args.consistency_max_valid_ratio
                    and consistency_stats["skipped"] < 0.5
                ):
                    consistency_weight = args.lambda_consistency * min(
                        1.0,
                        max(0.0, (iteration - consistency_start + 1) / max(1, args.consistency_ramp_iters)),
                    )
                    loss += consistency_weight * loss_mat_ref_gs
                    loss_consist = loss_mat_ref_gs
                else:
                    consistency_weight = 0.0
                    loss_consist = torch.zeros((), device=loss.device)

                consistency_debug_logger.record(
                    iteration=iteration,
                    src_uid=int(getattr(viewpoint_src, "uid", -1)),
                    tgt_uid=consistency_tgt_uid,
                    rank=consistency_rank,
                    loss_value=float(loss_consist.detach().item()) if isinstance(loss_consist, torch.Tensor) else float(loss_consist),
                    weighted_loss=float(consistency_weight * (loss_consist.detach().item() if isinstance(loss_consist, torch.Tensor) else loss_consist)),
                    stats=consistency_stats,
                    min_valid_ratio=args.consistency_min_valid_ratio,
                    max_valid_ratio=args.consistency_max_valid_ratio,
                )

            if iteration % 100 == 0:
                save_pbr_debug_montage(
                    rendering_result=rendering_result,
                    gt_image=gt_image,
                    iteration=iteration,
                    save_dir=os.path.join(args.model_path, "debug_pbr_montage")
                )

            #### envmap
            # TV smoothness
            envmap = dr.texture(
                cubemap.base[None, ...],
                envmap_dirs[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )[
                0
            ]  # [H, W, 3]
            tv_h1 = torch.pow(envmap[1:, :, :] - envmap[:-1, :, :], 2).mean()
            tv_w1 = torch.pow(envmap[:, 1:, :] - envmap[:, :-1, :], 2).mean()
            env_tv_loss = tv_h1 + tv_w1
            loss += env_tv_loss * env_tv_weight

            # =========================================================
            # [New Feature] 光照正则化 (Light Regularization / Neutral Prior)
            # =========================================================
            # 只有当开启参数且进入 PBR 阶段时执行
            if args.use_light and +args.lambda_light > 0:
                # 1. 获取当前采样的环境光颜色 [H, W, 3]
                # 注意：这里直接复用你代码里已经算好的 envmap 变量
                # envmap = dr.texture(...) # 确保这行在你插入代码之前已经运行了

                if 'envmap' in locals():
                    # 2. 计算每个像素的 RGB 均值 (即亮度/灰度) -> [H, W, 1]
                    mean_intensity = envmap.mean(dim=-1, keepdim=True)

                    # 3. 计算每个通道与均值的偏差 (L1 距离)
                    # 这一步就是图片里公式的工程实现：让 R, G, B 都去接近 Mean
                    color_bias = torch.abs(envmap - mean_intensity)

                    # 4. 求平均得到 Loss
                    loss_light_reg = color_bias.mean()

                    # 5. 加到总 Loss 里
                    loss += loss_light_reg * args.lambda_light

            # =========================================================

            # # 【重点在这里】
            # # 假设 30000 步之前是纯粹的 3DGS 几何拟合阶段
            # # 在第 30000 步时，几何（形状和旋转）已经基本成型
            # if iteration == 30000:
            #     # 使用最短轴完全覆盖初始化法线
            #     gaussians.init_normal(coe=0.0)
            #     print("Initialized learned normals with shortest axis!")
            #
            # # 有些策略也会在 30000 步之后，每隔一定的步数进行一次“软混合”，防止法线跑偏
            # elif iteration > 30000 and iteration % 1001 == 0:
            #     # 将学习到的法线和当前的最短轴法线按比例混合（比如 0.5）
            #     gaussians.init_normal(coe=0.0)

        #在这下面写的不管stage1还是stage2都生效
        # ==========================================================
        # [新增] 尺度正则化 (Anisotropic Scale Penalty)
        # ==========================================================
        if args.lambda_scale_reg > 0.0:
            scales = gaussians.get_scaling  # 获取所有高斯的真实尺度参数 [N, 3]

            # 找到每个高斯的最长轴和最短轴
            max_scales = torch.max(scales, dim=1).values
            min_scales = torch.min(scales, dim=1).values

            # 计算各向异性比例 (长轴 / 短轴)
            # 加上 1e-7 防止除以零导致 NaN
            scale_ratio = max_scales / (min_scales + 1e-7)

            # 使用 ReLU 设定一个阈值（Margin）
            # 我们允许高斯变成椭球，但不允许极端的形变（比例 > 10.0）
            # 只有比例超过 10.0 的高斯才会产生 Loss 惩罚
            scale_loss = torch.mean(torch.nn.functional.relu(scale_ratio - 10.0))

            # 将正则化 Loss 加入总 Loss
            loss = loss + args.lambda_scale_reg * scale_loss
        # ==========================================================

        loss.backward()
        # print("back")

        with torch.no_grad():
            current_train_psnr = psnr(train_render_for_metrics.clamp(0.0, 1.0), gt_image.clamp(0.0, 1.0)).mean().item()
            if iteration > pbr_iteration:
                normal_prop_debug_logger.record(
                    iteration=iteration,
                    active=normal_prop_active,
                    train_psnr=current_train_psnr,
                    loss_value=loss.detach().item(),
                    pbr_l1=pbr_render_loss_for_debug,
                    stats=normal_prop_stats,
                    grad_stats=normal_prop_grad_stats,
                    lr_stats={
                        "normal": get_optimizer_group_lr(gaussians.optimizer, "normal"),
                        "material": get_optimizer_group_lr(gaussians.optimizer, "albedo"),
                        "light": get_optimizer_first_lr(light_optimizer),
                    },
                )

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                # 1. 准备 loss 数值
                if isinstance(normal_loss, torch.Tensor):
                    normal_loss_val = normal_loss.item()
                else:
                    normal_loss_val = normal_loss

                if isinstance(loss_mono_depth, torch.Tensor):
                    loss_mono_depth_val = loss_mono_depth.item()
                else:
                    loss_mono_depth_val = loss_mono_depth

                if isinstance(loss_omnidata, torch.Tensor):
                    loss_omnidata_val = loss_omnidata.item()
                else:
                    loss_omnidata_val = loss_omnidata


                loss_log = {
                    "Loss": f"{loss.item():.{5}f}",
                    "Org_N": f"{normal_loss_val:.{5}f}"
                }

                # 2. 如果开启深度先验，往字典里加一项
                if use_mono_depth:
                    loss_log["New_D"] = f"{loss_mono_depth_val:.{5}f}"
                if use_mono_normal:
                    loss_log["New_N"] = f"{loss_omnidata_val:.{5}f}"


                # 【新增】获取一致性 Loss 的数值
                # 因为你设置了每 5 步算一次，没算的时候它就是 0.0
                if isinstance(loss_consist, torch.Tensor):
                    loss_consist_val = loss_consist.item()
                else:
                    loss_consist_val = loss_consist
                # 只有当开启了功能，且进入了 PBR 阶段，且数值不为 0 时才显示
                if args.use_consistency and iteration > pbr_iteration:
                    loss_log["Con"] = f"{loss_consist_val:.{5}f}"
                    loss_log["ConV"] = f"{consistency_stats['valid_ratio']:.{3}f}"
                if getattr(args, "use_normal_propagation", False) and normal_prop_active:
                    loss_log["NProp"] = f"{normal_prop_valid_ratio:.{3}f}"
                    loss_log["NPskip"] = f"{int(normal_prop_stats['skipped'])}"
                # =========================================================

                # 3. 更新进度条 (注意这里的缩进，必须在 if use_mono_depth 外面)
                progress_bar.set_postfix(loss_log)
                progress_bar.update(10)

                #额外打印，用于回看
                # if iteration % 500 == 0:
                #     # 格式：[Iter 1500] Loss: 0.12345 | Org_N: 0.05000 | New_D: 0.02000
                #     print(
                #         f"\n[Iter {iteration:05d}] "
                #         f"Loss: {loss.item():.5f} | "
                #         f"Org_N: {normal_loss_val:.5f} | "
                #         f"New_D: {loss_mono_depth_val:.5f}"
                #         f"New_N: {loss_omnidata_val:.5f}"
                #     )

                # === 【新增】记录 Loss 曲线到 TensorBoard ===
                if tb_writer is not None:
                    # 记录总 Loss
                    tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

                    # 记录原始几何 Loss
                    tb_writer.add_scalar('train_loss_patches/original_normal_loss', normal_loss_val, iteration)

                    # 记录单目深度监督 Loss (如果开启)
                    if use_mono_depth:
                        tb_writer.add_scalar('train_loss_patches/mono_depth_loss', loss_mono_depth_val, iteration)

                    if args.use_consistency and iteration > pbr_iteration:
                        tb_writer.add_scalar('train_loss_patches/consist_loss', loss_consist.item() if isinstance(loss_consist, torch.Tensor) else loss_consist, iteration)
                        tb_writer.add_scalar('train_loss_patches/consist_valid_ratio', consistency_stats["valid_ratio"], iteration)
                        tb_writer.add_scalar('train_loss_patches/consist_occ_ratio', consistency_stats["occ_ratio"], iteration)
                        tb_writer.add_scalar('train_loss_patches/consist_edge_ratio', consistency_stats["edge_ratio"], iteration)
                        tb_writer.add_scalar('train_loss_patches/consist_rough_l1', consistency_stats["rough_l1"], iteration)
                        tb_writer.add_scalar('train_loss_patches/consist_metal_l1', consistency_stats["metal_l1"], iteration)
                        tb_writer.add_scalar('train_loss_patches/consist_skipped', consistency_stats["skipped"], iteration)
                    if getattr(args, "use_normal_propagation", False) and normal_prop_active:
                        tb_writer.add_scalar('train_loss_patches/normal_prop_valid_ratio', normal_prop_valid_ratio, iteration)
                        tb_writer.add_scalar('train_loss_patches/normal_prop_image_ratio', normal_prop_stats["image_ratio"], iteration)
                        tb_writer.add_scalar('train_loss_patches/normal_prop_rough_ratio', normal_prop_stats["rough_ratio"], iteration)
                        tb_writer.add_scalar('train_loss_patches/normal_prop_spec_ratio', normal_prop_stats["spec_ratio"], iteration)
                        tb_writer.add_scalar('train_loss_patches/normal_prop_skipped', normal_prop_stats["skipped"], iteration)
                        tb_writer.add_scalar('train_loss_patches/normal_prop_grad_raw_norm', normal_prop_grad_stats["raw"], iteration)
                        tb_writer.add_scalar('train_loss_patches/normal_prop_grad_masked_norm', normal_prop_grad_stats["masked"], iteration)
                #==========================================

            if iteration in saving_iterations:
                print(f"\n[INFO] Saving Gaussian model at iteration {iteration}...")
                scene.save(iteration)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer=tb_writer,
                iteration=iteration,
                Ll1=Ll1,
                normal_loss=normal_loss,
                loss=loss,
                elapsed=0,
                testing_iterations=testing_iterations,
                scene=scene,
                light=cubemap,
                brdf_lut=brdf_lut,
                canonical_rays=canonical_rays,
                pbr_iteration=pbr_iteration,
                metallic=metallic,
                tone=tone,
                gamma=gamma,
                radius=radius,
                bias=bias,
                thick=thick,
                delta=delta,
                step=step,
                start=start,
                renderArgs=(pipe, background),
                indirect=indirect,
            )
            # NOTE: we same .pth instead of point cloud for additional irradiance volumes and cubemap
            # if iteration in saving_iterations:
            #    print(f"\n[ITER {iteration}] Saving Gaussians")
            #    scene.save(iteration)
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    {
                        "gaussians": gaussians.capture(),
                        "cubemap": cubemap.state_dict(),
                        # "irradiance_volumes": irradiance_volumes.state_dict(),
                        "light_optimizer": light_optimizer.state_dict(),
                        "iteration": iteration,
                    },
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    {
                        "gaussians": gaussians.capture(),
                        "cubemap": cubemap.state_dict(),
                        # "irradiance_volumes": irradiance_volumes.state_dict(),
                        "light_optimizer": light_optimizer.state_dict(),
                        "iteration": iteration,
                    },
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(iteration)
                if iteration >= pbr_iteration:
                    light_optimizer.step()
                    light_optimizer.zero_grad(set_to_none=True)
                    cubemap.clamp_(min=0.0)
            # ---------------- 颜色破坏策略 (Color Sabotage PBR版) ----------------
            if sabotage_active and (iteration % args.sabotage_interval == 0):
                # 1. 获取当前所有高斯的粗糙度
                # 注意：这里请替换为您代码中实际获取粗糙度的方法，如 gaussians.get_roughness()
                roughness = gaussians.get_roughness

                # 2. 统计当前“反射高斯”的数量 (粗糙度 < 阈值，说明它正在拟合高光)
                # 阈值可以稍微设低一点以确保它是真正的反射面，这里使用 1.0 - rough_thresh 作为对称概念
                current_specular_count = (roughness < args.sabotage_rough_thresh).sum().item()

                # 3. 终止条件检查：如果"反射高斯"数量还在涨，说明策略还在发掘新的反射面
                if current_specular_count > max_specular_count:
                    max_specular_count = current_specular_count
                    sabotage_patience_counter = 0  # 重置耐心值
                else:
                    sabotage_patience_counter += 1

                if sabotage_patience_counter >= args.sabotage_patience:
                    print(f"[Iteration {iteration}] 颜色破坏策略终止：低粗糙度(反射)高斯数量稳定。")
                    sabotage_active = False
                else:
                    # 4. 执行颜色破坏：打乱那些高粗糙度高斯的颜色，逼迫网络用粗糙度/法线去拟合高光
                    gaussians.apply_color_sabotage(
                        roughness,
                        rough_threshold=args.sabotage_rough_thresh,
                        noise_level=args.sabotage_noise
                    )

        # time.sleep(0.15)
        torch.cuda.empty_cache()


def prepare_output_and_logger(args: GroupParams) -> Optional[SummaryWriter]:
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer: Optional[SummaryWriter],
    iteration: int,
    Ll1: Union[float, torch.Tensor],
    normal_loss: Union[float, torch.Tensor],
    loss: Union[float, torch.Tensor],
    elapsed: float,
    testing_iterations: List[int],
    scene: Scene,
    light: CubemapLight,
    brdf_lut: torch.Tensor,
    canonical_rays: torch.Tensor,
    pbr_iteration: int,
    metallic: bool,
    tone: bool,
    gamma: bool,
    radius: float,
    bias: float,
    thick: float,
    delta: float,
    step: int,
    start: int,
    renderArgs: Tuple[GroupParams, torch.Tensor],
    indirect: bool = False,
) -> None:
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1, iteration)
        tb_writer.add_scalar("train_loss_patches/normal_loss", normal_loss, iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss, iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )
        lpips_fn = LPIPS(net="vgg").cuda()
        pipe, background = renderArgs
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    viewpoint: Camera
                    render_result = render(
                        viewpoint_camera=viewpoint,
                        pc=scene.gaussians,
                        pipe=pipe,
                        bg_color=background,
                        inference=True,
                        pad_normal=False,
                        derive_normal=True,
                        radius=radius,
                        bias=bias,
                        thick=thick,
                        delta=delta,
                        step=step,
                        start=start)

                    tanfovx = math.tan(viewpoint.FoVx * 0.5)
                    tanfovy = math.tan(viewpoint.FoVy * 0.5)
                    image_height=int(viewpoint.image_height)
                    image_width=int(viewpoint.image_width)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    depth_img = (
                        torch.from_numpy(
                            turbo_cmap(render_result["depth_map"].cpu().numpy().squeeze())
                        )
                        .to(image.device)
                        .permute(2, 0, 1)
                    )
                    normal_map_from_depth = render_result["normal_map_from_depth"]
                    normal_map = render_result["normal_map"]
                    normal_img = torch.cat([normal_map, normal_map_from_depth], dim=-1)
                    # gt_image = viewpoint.original_image.cuda()
                    gt_image = viewpoint.original_image[0:3, :, :].cuda()
                    alpha_mask = viewpoint.gt_alpha_mask.cuda()
                    gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
                    albedo_map = render_result["albedo_map"]  # [3, H, W]
                    roughness_map = render_result["roughness_map"]  # [1, H, W]
                    metallic_map = render_result["metallic_map"]  # [1, H, W]
                    out_normal_view = render_result["out_normal_view"]
                    depth_pos = render_result["depth_pos"]
                    normal_mask = render_result["normal_mask"]  # [1, H, W]
                    brdf_map = torch.cat(
                        [
                            albedo_map,
                            torch.tile(roughness_map, (3, 1, 1)),
                            torch.tile(metallic_map, (3, 1, 1)),
                        ],
                        dim=2,
                    )  # [3, H, 3W]
                    # NOTE: PBR record
                    if iteration > pbr_iteration:
                        H, W = viewpoint.image_height, viewpoint.image_width
                        c2w = torch.inverse(viewpoint.world_view_transform.T)  # [4, 4]
                        view_dirs = -(
                            (
                                F.normalize(canonical_rays[:, None, :], p=2, dim=-1)
                                * c2w[None, :3, :3]
                            )  # [HW, 3, 3]
                            .sum(dim=-1)
                            .reshape(H, W, 3)
                        )  # [H, W, 3]
                        normal_mask = render_result["normal_mask"]

                        # recon occlusion
                        if indirect:
                            occlusion = render_result["occlusion_map"].permute(1, 2, 0)
                        else:
                            occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]

                        # build mip for environment light
                        light.build_mips()
                        pbr_result = pbr_shading(
                            light=light,
                            normals=normal_map.permute(1, 2, 0),  # [H, W, 3]
                            view_dirs=view_dirs,
                            mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
                            albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
                            roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
                            metallic=metallic_map.permute(1, 2, 0)
                            if metallic
                            else None,  # [H, W, 1]
                            tone=tone,
                            gamma=gamma,
                            brdf_lut=brdf_lut,
                            occlusion=occlusion
                        )

                        diffuse_rgb = (
                            pbr_result["diffuse_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                        )  # [3, H, W]
                        # specular_rgb = (
                        #     pbr_result["specular_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                        # )  # [3, H, W]
                        # NOTE: mask render_rgb by depth map
                        background = renderArgs[1]
                        diffuse_rgb = torch.where(
                            normal_mask,
                            diffuse_rgb,
                            background[:, None, None],
                        )
                        # specular_rgb = torch.where(
                        #     normal_mask,
                        #     specular_rgb,
                        #     background[:, None, None],
                        # )
                        pbr_image = torch.cat(
                            [diffuse_rgb, diffuse_rgb], dim=2
                        )  # [3, H, 3W]

                        # linear_rgb = (
                        #     pbr_result["linear_rgb"].permute(2, 0, 1)
                        # )

                        # linear_rgb = torch.where(
                        #     normal_mask,
                        #     linear_rgb,
                        #     background[:, None, None]
                        # )
                        render_direct = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
                        render_direct = torch.where(
                        normal_mask,
                        render_direct,
                        background[:, None, None])

                        SSR = Gaussian_SSR(tanfovx, tanfovy, image_width, image_height, radius, bias, thick, delta, step, start)
                        if metallic:
                            F0 = (1.0 - metallic_map) * 0.04 + albedo_map * metallic_map
                        else:
                            F0 = torch.ones_like(albedo_map) * 0.04  # [1, H, W, 3]
                            metallic_map = torch.zeros_like(roughness_map)

                        linear_rgb = srgb_to_linear(render_direct)
                        (IRR, _) = SSR(out_normal_view, depth_pos, linear_rgb, albedo_map, roughness_map, metallic_map, F0)
                        IRR = linear_to_srgb(IRR)
                        IRR = kornia.filters.median_blur(IRR[None, ...], (3, 3))[0]
                        render_rgb = render_direct + IRR
                        render_rgb = torch.where(
                            normal_mask,
                            render_rgb,
                            background[:, None, None])
                    else:
                        zero_pad = torch.zeros_like(image)
                        render_rgb = zero_pad
                        pbr_image = torch.cat([zero_pad, zero_pad, zero_pad], dim=2)  # [3, H, 3W]

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}_{idx}/render",
                            resize_tensorboard_img(image)[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}_{idx}/depth",
                            resize_tensorboard_img(depth_img)[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}_{idx}/normal",
                            (resize_tensorboard_img(normal_img, 1600)[None] + 1.0) / 2.0,
                            global_step=iteration,
                        )
                        if iteration > pbr_iteration:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}_{idx}/brdf",
                                resize_tensorboard_img(brdf_map, 2400)[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}_{idx}/pbr_render",
                                resize_tensorboard_img(pbr_image, 2400)[None],
                                global_step=iteration,
                            )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}_{idx}/ground_truth",
                                resize_tensorboard_img(gt_image)[None],
                                global_step=iteration,
                            )
                    if iteration > pbr_iteration:
                        l1_test += F.l1_loss(render_rgb, gt_image).mean().double()
                        psnr_test += psnr(render_rgb, gt_image).mean().double()
                        ssim_test += ssim(render_rgb, gt_image).mean().double()
                        lpips_test += lpips_fn(render_rgb, gt_image).mean().double()
                    else:
                        l1_test += F.l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                lpips_test /= len(config["cameras"])
                print(len(config["cameras"]))
                # print(
                #     f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.6f} PSNR {psnr_test:.6f} SSIM {ssim_test:.6f} LPIPS {lpips_test:.6f}"
                # )

                # --- 修改开始：写入指标和运行命令 ---
                # 1. 准备评估结果字符串
                eval_log = f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.6f} PSNR {psnr_test:.6f} SSIM {ssim_test:.6f} LPIPS {lpips_test:.6f}"
                print(eval_log)  # 打印到控制台

                # 2. 获取当前运行的完整命令
                # sys.argv 包含了脚本名和所有参数，我们用空格连接起来
                current_cmd = "python " + " ".join(sys.argv)

                # 3. 写入 metrics.txt
                metrics_filepath = os.path.join(scene.model_path, "metrics.txt")
                with open(metrics_filepath, "a") as f:
                    f.write(eval_log)
                    f.write(f"\n[Command] {current_cmd}\n")  # 写入命令
                    f.write("-" * 50)  # 添加分割线，方便区分多次评估

                print(f"[INFO] Metrics and command saved to {metrics_filepath}")
                # --- 修改结束 ---

                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity.reshape(-1), iteration
            )
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30000, 32_000, 33000, 34000, 35000, 36000, 37000, 38000, 39000],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30000, 32_000, 33000, 34000, 35000, 36000, 37000, 38000, 39000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--pbr_iteration", default=30_000, type=int, help="The iteration to begin the pb.r learning (Deomposition Stage in the paper)")
    parser.add_argument("--normal_tv", default=5.0, type=float, help="The weight of TV loss on predicted normal map.")
    parser.add_argument("--brdf_tv", default=1.0, type=float, help="The weight of TV loss on predicted BRDF (material) map.")
    parser.add_argument("--env_tv", default=0.01, type=float, help="The weight of TV loss on Environment Map.")
    parser.add_argument("--radius", default=0.8, type=float, help="Path tracing range")
    parser.add_argument("--bias", default=0.01, type=float, help="ensure hit the surface")
    parser.add_argument("--thick", default=0.05, type=float, help="thickness of the surface")
    parser.add_argument("--delta", default=0.0625, type=float, help="angle interval to control the num-sample")
    parser.add_argument("--step", default=16, type=int, help="Path tracing steps")
    parser.add_argument("--start", default=8, type=int, help="Path tracing starting point")
    parser.add_argument("--degree", default=3, type=int, help="sh_degree")
    parser.add_argument("--tone", action="store_true", help="Enable aces film tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear_to_sRGB for gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Enable metallic material reconstruction.")
    parser.add_argument("--indirect", action="store_true", help="Enable indirect diffuse modeling.")
    parser.add_argument("--use_normal_propagation", action="store_true",
                        help="Enable staged, masked reflection-gradient normal propagation during PBR training.")
    parser.add_argument("--normal_prop_start", default=0, type=int,
                        help="Absolute iteration to start normal propagation. <=0 starts right after --pbr_iteration.")
    parser.add_argument("--normal_prop_iters", default=8000, type=int,
                        help="Number of iterations to run normal propagation after normal_prop_start.")
    parser.add_argument("--normal_prop_rough_thresh", default=0.35, type=float,
                        help="Pixels with roughness below this threshold receive propagated normal gradients.")
    parser.add_argument("--normal_prop_spec_thresh", default=0.03, type=float,
                        help="Pixels with specular intensity above this threshold receive propagated normal gradients.")
    parser.add_argument("--normal_prop_normal_lr", default=0.005, type=float,
                        help="Low learning rate for Gaussian normal parameters during propagation.")
    parser.add_argument("--normal_prop_normal_grad_scale", default=1.0, type=float,
                        help="Additional scale for masked normal gradients during propagation.")
    parser.add_argument("--normal_prop_depth_weight", default=0.05, type=float,
                        help="Weak normal-from-depth stabilizer added during normal propagation.")
    parser.add_argument("--normal_prop_material_lr_scale", default=0.1, type=float,
                        help="LR multiplier for albedo/roughness/metallic params during propagation; 0 freezes optimizer updates.")
    parser.add_argument("--normal_prop_material_grad_scale", default=0.1, type=float,
                        help="Gradient multiplier for albedo/roughness/metallic maps during propagation; 0 freezes PBR material gradients.")
    parser.add_argument("--normal_prop_light_lr_scale", default=0.25, type=float,
                        help="LR multiplier for environment light optimizer during propagation.")
    parser.add_argument("--normal_prop_mask_mode", default="rough_and_spec",
                        choices=["rough_and_spec", "rough_or_spec", "rough_only", "spec_only"],
                        help="Reliability mask used for normal propagation. rough_and_spec is the safest default; rough_or_spec matches the original permissive behavior.")
    parser.add_argument("--normal_prop_min_valid_ratio", default=0.001, type=float,
                        help="Skip normal propagation for a view if the selected valid-mask ratio is below this value.")
    parser.add_argument("--normal_prop_max_valid_ratio", default=0.25, type=float,
                        help="Skip normal propagation for a view if the selected valid-mask ratio is above this value; high values usually mean an over-broad noisy mask.")
    parser.add_argument("--normal_prop_ramp_iters", default=1000, type=int,
                        help="Linearly ramp masked normal-gradient scale over this many propagation iterations.")
    parser.add_argument("--normal_prop_debug_interval", default=500, type=int,
                        help="Save normal-propagation debug montages every N active iterations; <=0 disables image dumps.")
    parser.add_argument("--normal_prop_save_individual_images", action="store_true",
                        help="Also save separate PNGs for each normal-propagation debug layer beside the montage.")
    parser.add_argument("--normal_prop_log_interval", default=50, type=int,
                        help="Write CSV/JSON file diagnostics every N iterations; <=0 disables file diagnostics.")
    parser.add_argument("--normal_prop_advice_drop_psnr", default=0.5, type=float,
                        help="Emit an automatic diagnosis when active PSNR EMA drops this many dB below the pre-propagation reference EMA.")
    #ljx:单目深度 损失函数权重参数
    # 【新增】位置约束开关
    parser.add_argument("--use_position_opt", action="store_true",
                        help="If True, optimize Gaussian positions using Depth L1/Log loss.")
    # 【新增】是否启用双边平滑 Loss
    parser.add_argument("--use_bilateral_loss", action="store_true",
                        help="Enable bilateral smoothness loss for roughness consistency.")
    # 【新增】双边平滑 Loss 的权重 (建议默认 0.1 - 1.0)
    parser.add_argument("--lambda_bilateral", default=0.1, type=float, help="Weight for bilateral smoothness loss.")
    # 【新增】边缘敏感度 (建议默认 10.0)
    parser.add_argument("--bilateral_edge", default=10.0, type=float,
                        help="Edge sensitivity for bilateral loss. Higher keeps edges sharper.")
    # 【新增】HSV Loss
    parser.add_argument("--lambda_hsv", default=0.0, type=float, help="Weight for HSV decoupling loss.")
    # 【新增】高光自适应掩码
    parser.add_argument("--use_specular_mask", action="store_true",
                        help="Enable self-supervised specular masking to prevent albedo baking.")
    # [New Feature] 多视角重投影一致性参数
    parser.add_argument("--use_consistency", action="store_true",
                        help="Enable multi-view reprojection consistency loss.")
    parser.add_argument("--lambda_consistency", default=0.05, type=float,
                        help="Weight for consistency loss. Suggest starting from 0.01 - 0.05; the previous 0.1 can over-regularize materials.")
    parser.add_argument("--consistency_start", default=0, type=int,
                        help="Absolute iteration to start multi-view material consistency. <=0 uses pbr_iteration + consistency_start_offset.")
    parser.add_argument("--consistency_start_offset", default=2000, type=int,
                        help="Delay after pbr_iteration before enabling consistency when consistency_start <= 0.")
    parser.add_argument("--consistency_interval", default=5, type=int,
                        help="Compute multi-view consistency every N iterations to reduce noisy gradients and cost.")
    parser.add_argument("--consistency_ramp_iters", default=1000, type=int,
                        help="Linearly ramp consistency weight over this many active iterations.")
    parser.add_argument("--consistency_rank_min", default=1, type=int,
                        help="Minimum spatial-KNN neighbor rank for target view sampling; 1 is nearest non-self view.")
    parser.add_argument("--consistency_rank_max", default=4, type=int,
                        help="Maximum spatial-KNN neighbor rank for target view sampling.")
    parser.add_argument("--consistency_min_valid_ratio", default=0.002, type=float,
                        help="Skip consistency if reprojected valid mask ratio is below this value.")
    parser.add_argument("--consistency_max_valid_ratio", default=0.5, type=float,
                        help="Skip consistency if reprojected valid mask ratio is above this value, which usually indicates an over-broad unreliable mask.")
    parser.add_argument("--consistency_occ_abs_thresh", default=0.02, type=float,
                        help="Absolute depth agreement threshold for reprojection occlusion filtering.")
    parser.add_argument("--consistency_occ_rel_thresh", default=0.01, type=float,
                        help="Relative depth agreement threshold for reprojection occlusion filtering.")
    parser.add_argument("--consistency_edge_rel_thresh", default=0.03, type=float,
                        help="Relative source-depth edge threshold; lower values reject more depth discontinuities.")
    parser.add_argument("--consistency_facing_thresh", default=0.02, type=float,
                        help="Minimum normal dot target-view direction for consistency pixels.")
    parser.add_argument("--consistency_robust_eps", default=1e-3, type=float,
                        help="Charbonnier epsilon for robust material consistency loss.")
    parser.add_argument("--consistency_albedo", action="store_true",
                        help="Also constrain albedo consistency; disabled by default to avoid baking lighting errors.")
    parser.add_argument("--consistency_debug_interval", default=250, type=int,
                        help="Save multi-view consistency mask/error montages every N iterations; <=0 disables image dumps.")
    parser.add_argument("--consistency_log_interval", default=50, type=int,
                        help="Write CSV/JSON consistency diagnostics every N iterations; <=0 disables file logging.")

    # [新增] 开关中性光loss
    # action='store_true' 表示：只要命令行里写了 --use_light，这个值就是 True，否则是 False
    parser.add_argument("--use_light", action='store_true', default=False,
                        help="Switch to enable light optimization/regularization module")
    # 权重参数
    parser.add_argument("--lambda_light", type=float, default=0.01, help="Weight for light regularization")

    # 在您的 OptimizationParams 类或 argparse 设置中添加：
    # 颜色破坏策略 (Color Sabotage) 专用参数
    parser.add_argument("--color_sabotage", action="store_true", help="是否启用基于粗糙度的颜色破坏策略")
    parser.add_argument("--sabotage_interval", type=int, default=100, help="执行颜色破坏的迭代间隔")
    parser.add_argument("--sabotage_noise", type=float, default=0.1, help="加入的基础颜色噪声比例 (+-10%)")
    # PBR 适配：粗糙度大于此值被视为“尚未成为反射体”的高斯
    parser.add_argument("--sabotage_rough_thresh", type=float, default=0.6, help="执行颜色破坏的粗糙度下限")
    parser.add_argument("--sabotage_patience", type=int, default=5, help="当反射高斯数量不再增加时终止策略的容忍次数")
    #尺度正则化
    parser.add_argument("--lambda_scale_reg", type=float, default=0.0,
                        help="Weight for anisotropic scale penalty (0.0 to disable)")

    args = parser.parse_args(sys.argv[1:])
    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)


    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # with torch.autograd.detect_anomaly():
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    dataset.use_mono_depth = args.use_mono_depth
    dataset.sh_degree = args.degree
    training(
        dataset=dataset,
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint_path=args.start_checkpoint,
        pbr_iteration=args.pbr_iteration,
        debug_from=args.debug_from,
        metallic=args.metallic,
        tone=args.tone,
        gamma=args.gamma,
        normal_tv_weight=args.normal_tv,
        brdf_tv_weight=args.brdf_tv,
        env_tv_weight=args.env_tv,
        radius=args.radius,
        bias=args.bias,
        thick=args.thick,
        delta=args.delta,
        step=args.step,
        start=args.start,
        indirect=args.indirect,
        # 【新增】接收位置约束参数
        use_position_opt=args.use_position_opt,

    )

    # All done
