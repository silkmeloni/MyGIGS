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
from typing import Optional
from PIL import Image          # 用于 Image.open
import torch.nn.functional as F # 用于 F.interpolate
import numpy as np
import torch
from torch import nn
import cv2

from utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera(nn.Module):
    # ... 在 Camera 类内部 ...

    def get_calib_matrix_nerf(self, scale=1.0):
        """
        获取 NeRF 格式的内参和外参矩阵，用于法线渲染等计算。
        """
        # 根据 FoV 计算焦距
        # 注意：这里基于当前的图像分辨率 self.image_width/height
        # 如果传入了 scale，则对分辨率进行缩放
        W = self.image_width * scale
        H = self.image_height * scale

        # fx = W / (2 * tan(fovx / 2))
        focal_x = W / (2.0 * np.tan(self.FoVx * 0.5))
        focal_y = H / (2.0 * np.tan(self.FoVy * 0.5))

        # 假设主点在图像中心
        cx = W / 2.0
        cy = H / 2.0

        # 构建内参矩阵 (3x3)
        intrinsic_matrix = torch.tensor([
            [focal_x, 0, cx],
            [0, focal_y, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.data_device)

        # 获取外参矩阵 (World-to-Camera)
        # 3DGS 存储的 world_view_transform 通常是转置过的 (Column-major)，我们需要转回来
        extrinsic_matrix = self.world_view_transform.transpose(0, 1)

        return intrinsic_matrix, extrinsic_matrix

    def __init__(
        self,
        colmap_id: int,
        R: np.ndarray,
        T: np.ndarray,
        FoVx: float,
        FoVy: float,
        image: torch.Tensor,
        image_name: str,
        uid: int,
        gt_alpha_mask: Optional[torch.Tensor] = None,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        data_device: str = "cuda",
        depth_mono_path=None, #ljx
        normal_mono_path=None,
        depth_scale: float = 1.0,
        depth_shift: float = 0.0,
    ) -> None:
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # NOTE: OOM in develop machine, do not put them on GPU
        if gt_alpha_mask is not None:
            self.gt_alpha_mask = gt_alpha_mask
        else:
            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width))
        # if gt_alpha_mask is not None:
        #     self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones(
        #         (1, self.image_height, self.image_width), device=self.data_device
        #     )

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        #ljx:确保 Camera 类在初始化时读取深度图并将其上传到GPU
        self.depth_mono_path = depth_mono_path
        self.mono_depth_image = None

        #实际上是视差图
        if depth_mono_path is not None:
            # 1. 【无损读取】使用 -1 (IMREAD_UNCHANGED) 读取原始数据
            # 这样读出来的可能是 uint16 (0-65535) 也可能是 float32 (0.0-1.0)
            mono_depth_np = cv2.imread(depth_mono_path, cv2.IMREAD_UNCHANGED)

            # 容错：如果 cv2 读失败 (有些特殊压缩的 tiff)，尝试 tifffile
            if mono_depth_np is None:
                try:
                    import tifffile
                    mono_depth_np = tifffile.imread(depth_mono_path)
                except ImportError:
                    pass

            if mono_depth_np is not None:
                # 2. 处理多通道 (取第一通道)
                if mono_depth_np.ndim == 3:
                    mono_depth_np = mono_depth_np[:, :, 0]

                # 3. 转换为 float32
                mono_depth_np = mono_depth_np.astype(np.float32)

                # 4. 【智能归一化】
                # 你的 Log 显示数值已经是 0.18... (说明可能是 float)
                # 必须判断原始范围，防止 "Double Normalization" (本来是 0.2，又除以 255 变成 0.0007)

                max_val = mono_depth_np.max()
                if max_val > 255.0:
                    # 肯定是 16-bit 整数，归一化
                    mono_depth_np /= 65535.0
                elif max_val > 1.0:
                    # 肯定是 8-bit 整数 (0-255)，归一化
                    mono_depth_np /= 255.0
                # else: 如果 max_val <= 1.0，说明已经是 float (0-1)，啥都不要做！<-- 你之前的代码可能这里除了两次

                # 5. 转 Tensor
                mono_depth = torch.from_numpy(mono_depth_np).float().to(self.data_device)

                # 6. 【输入去量化 (Dithering)】 <--- 关键！
                # 既然数据本质是 8-bit (步长 1/255)，我们加上 1/255 的均匀噪声。
                # 这不是滤波，这是让平坦的台阶变成有坡度的斜面，解决 banding，同时不损失高频细节。
                if max_val > 1.0:  # 仅对整数源数据做这个操作
                    noise = (torch.rand_like(mono_depth) - 0.5) / 255.0
                    mono_depth = mono_depth + noise

                # 7. Resize (必须用 Bilinear 且 align_corners=False)
                if mono_depth.shape[0] != self.image_height or mono_depth.shape[1] != self.image_width:
                    mono_depth = mono_depth.unsqueeze(0).unsqueeze(0)
                    mono_depth = F.interpolate(
                        mono_depth,
                        size=(self.image_height, self.image_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)

                self.mono_depth_image = mono_depth
            else:
                self.mono_depth_image = None
        else:
            self.mono_depth_image = None

        # if depth_mono_path is not None:
        #     # 1. 读取原始数据
        #     mono_depth = Image.open(depth_mono_path)
        #     mono_depth = torch.from_numpy(np.array(mono_depth)).float().to(self.data_device)
        #
        #     # # 2. 归一化 (关键！必须和你的脚本保持一致)
        #     # # 脚本里写的是: invmonodepthmap.astype(np.float32) / (2**16)
        #     # # 假设输入的是 Disparity (Inverse Depth)
        #     # mono_disp = mono_depth / 65535.0
        #     #
        #     # # 3. Resize (如果 RGB 和 深度图 尺寸不一致)
        #     # if mono_disp.shape[0] != self.image_height or mono_disp.shape[1] != self.image_width:
        #     #     mono_disp = \
        #     #     F.interpolate(mono_disp[None, None, ...], size=(self.image_height, self.image_width), mode='bilinear',
        #     #                   align_corners=False)[0, 0]
        #     #
        #     # # 4. 应用预计算的 Scale 和 Shift
        #     # # 公式: Aligned_Disp = Scale * Mono_Disp + Shift
        #     # # 注意：这里的 scale/shift 已经是针对 dataset_readers 里读取的 float 值了
        #     # aligned_disp = mono_disp #* depth_scale + depth_shift
        #     #
        #     # # 5. 转换为 Metric Depth (用于后续训练监督)
        #     # aligned_disp = torch.clamp(aligned_disp, min=1e-7)
        #     #
        #     # # 最终存储的是真实的深度值 (Metric Depth)
        #     # self.mono_depth_image = 1.0 / aligned_disp
        #
        #     # 如果是RGBA或3通道，取第一通道
        #     if len(mono_depth.shape) == 3:
        #         mono_depth = mono_depth[:, :, 0]
        #         # Resize 到和 RGB 图像一样大 (如果需要)
        #     if mono_depth.shape[0] != self.image_height or mono_depth.shape[1] != self.image_width:
        #         mono_depth = \
        #             F.interpolate(mono_depth[None, None, ...], size=(self.image_height, self.image_width),
        #                           mode='bilinear',align_corners=False)[0, 0]
        #     self.mono_depth_image = mono_depth.to(data_device)
        # else:
        #     self.mono_depth_image = None

        # [新增] 读取单目法线 (使用的是切线空间法线，蓝紫色调，最后需要的也是切线空间  虽然名为world_normal但实际是切线空间）
        self.normal_mono_path = normal_mono_path
        self.mono_normal_image = None

        if normal_mono_path is not None:
            # 1. 读取图片并立即处理通道 (H, W, C)
            mono_normal_pil = Image.open(normal_mono_path)
            mono_normal_np = np.array(mono_normal_pil)

            # 【核心修复1】立刻切掉 Alpha 通道，只留 RGB
            # 防止 RGBA (4通道) 混入后续计算导致维度错误
            if mono_normal_np.shape[-1] == 4:
                mono_normal_np = mono_normal_np[:, :, :3]

            # 转 Tensor 并归一化到 [0, 1]
            mono_normal = torch.from_numpy(mono_normal_np).float() / 255.0

            # 2. Resize
            # 此时 mono_normal 是 [H, W, 3]
            if mono_normal.shape[0] != self.image_height or mono_normal.shape[1] != self.image_width:
                # permute 为 [1, 3, H, W] 进行插值
                mono_normal = mono_normal.permute(2, 0, 1).unsqueeze(0)
                mono_normal = F.interpolate(mono_normal, size=(self.image_height, self.image_width),
                                            mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

            # 3. 转换到 [-1, 1] 范围
            # 此时 mono_normal 依然是 [H, W, 3]
            mono_normal = mono_normal * 2.0 - 1.0

            # 4. 坐标系转换
            # 调整维度为 [3, H, W] 并放入 GPU
            mono_normal = mono_normal.permute(2, 0, 1).to(self.data_device)

            # 【核心修复2】保命逻辑：处理零向量/背景噪音
            # 如果像素是 (0,0,0)（也就是原图的中性灰或背景），归一化会除以0 -> NaN -> Loss爆炸
            norm = torch.norm(mono_normal, dim=0, keepdim=True)
            invalid_mask = norm < 1e-6
            # 将无效像素填充为默认法线 (比如 0,0,1)，防止计算出错
            if invalid_mask.any():
                mono_normal[0][invalid_mask[0]] = 0.0
                mono_normal[1][invalid_mask[0]] = 0.0
                mono_normal[2][invalid_mask[0]] = 1.0

            # 5. 安全归一化 (View Space)
            mono_normal = F.normalize(mono_normal, dim=0, eps=1e-6)

            # # 6. 旋转: View Space -> World Space
            # # R_wc = R_cw^T
            # R_wc = torch.tensor(self.R.T, device=self.data_device, dtype=torch.float32)
            #
            # # 锁定正确的 H, W
            C, H, W = mono_normal.shape

            # 矩阵乘法
            # 强制 reshape 为 (3, -1)，因为我们已经切掉了 Alpha，这里绝对安全
            mono_normal_flat = mono_normal.reshape(3, -1)
            #mono_normal_world = torch.matmul(R_wc, mono_normal_flat)
            mono_normal_world = mono_normal_flat

            # 7. 还原形状
            # 【核心修复3】不要用 orig_shape (它可能带了旧的错误维度)，直接用 H, W
            mono_normal_world = mono_normal_world.reshape(3, H, W)

            # 8. 最终归一化
            mono_normal_world = F.normalize(mono_normal_world, dim=0, eps=1e-6)

            self.mono_normal_image = mono_normal_world


class MiniCam:
    def __init__(
        self,
        width: int,
        height: int,
        fovy: float,
        fovx: float,
        znear: float,
        zfar: float,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
    ) -> None:
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]