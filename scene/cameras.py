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

        if depth_mono_path is not None:
            # 1. 读取原始数据
            # 你的脚本是用 cv2.IMREAD_UNCHANGED 读取 png，通常是 16bit 整数
            # 这里用 PIL 读取并转换
            mono_depth = Image.open(depth_mono_path)
            mono_depth = torch.from_numpy(np.array(mono_depth)).float().to(self.data_device)

            # 2. 归一化 (关键！必须和你的脚本保持一致)
            # 脚本里写的是: invmonodepthmap.astype(np.float32) / (2**16)
            # 假设输入的是 Disparity (Inverse Depth)
            mono_disp = mono_depth / 65535.0

            # 3. Resize (如果 RGB 和 深度图 尺寸不一致)
            if mono_disp.shape[0] != self.image_height or mono_disp.shape[1] != self.image_width:
                mono_disp = \
                F.interpolate(mono_disp[None, None, ...], size=(self.image_height, self.image_width), mode='bilinear',
                              align_corners=False)[0, 0]

            # 4. 应用预计算的 Scale 和 Shift
            # 公式: Aligned_Disp = Scale * Mono_Disp + Shift
            # 注意：这里的 scale/shift 已经是针对 dataset_readers 里读取的 float 值了
            aligned_disp = mono_disp * depth_scale + depth_shift

            # 5. 转换为 Metric Depth (用于后续训练监督)
            # 防止除以 0 或负数 (视差为0意味着无穷远)
            aligned_disp = torch.clamp(aligned_disp, min=1e-7)

            # 最终存储的是真实的深度值 (Metric Depth)
            self.mono_depth_image = 1.0 / aligned_disp
        else:
            self.mono_depth_image = None
        # if depth_mono_path is not None:
        #     # 读取深度图，通常单目深度是单通道
        #     # 注意：Depth Anything 输出通常是相对深度（disparity 或 inverse depth），或者是 metric depth
        #     # 这里假设读取为 float32 的 tensor
        #     mono_depth = Image.open(depth_mono_path)
        #     mono_depth = torch.from_numpy(np.array(mono_depth)).float()
        #
        #     # 如果是RGBA或3通道，取第一通道
        #     if len(mono_depth.shape) == 3:
        #         mono_depth = mono_depth[:, :, 0]
        #
        #     # Resize 到和 RGB 图像一样大 (如果需要)
        #     if mono_depth.shape[0] != self.image_height or mono_depth.shape[1] != self.image_width:
        #         mono_depth = \
        #         F.interpolate(mono_depth[None, None, ...], size=(self.image_height, self.image_width), mode='bilinear',
        #                       align_corners=False)[0, 0]
        #
        #     # 归一化处理 (建议归一化到 0-1 或标准化，这取决于你后面怎么对齐)
        #     # mono_depth = (mono_depth - mono_depth.min()) / (mono_depth.max() - mono_depth.min() + 1e-5)
        #
        #     self.mono_depth_image = mono_depth.to(data_device)


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