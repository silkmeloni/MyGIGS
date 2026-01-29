import torch
import torch.nn.functional as F
import torchvision
import os


# ==========================================
# 1. 基础辅助函数 (必须定义在 warp_consistency_loss 之前)
# ==========================================

def get_grid(H, W, device):
    """ 生成归一化的像素坐标网格 (H, W, 2) """
    y_range = torch.arange(H, dtype=torch.float32, device=device)
    x_range = torch.arange(W, dtype=torch.float32, device=device)
    Y, X = torch.meshgrid(y_range, x_range, indexing='ij')
    # [H, W, 2] -> (x, y)
    xy_grid = torch.stack([X, Y], dim=-1)
    return xy_grid


def depth_point_to_world(depth_map, camera):
    """
    将深度图反投影到世界坐标
    depth_map: [1, 1, H, W]
    camera: 3DGS 的 Camera 对象
    return: [1, 3, H, W] 世界坐标图
    """
    if depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(0)  # 确保是 [1, 1, H, W]

    H, W = depth_map.shape[2], depth_map.shape[3]
    device = depth_map.device

    # 1. 获取像素坐标
    grid = get_grid(H, W, device)  # [H, W, 2]
    u, v = grid[..., 0], grid[..., 1]

    # 2. 获取相机内参
    # fx = W / (2 * tan(fovx / 2))
    fx = W / (2 * torch.tan(torch.tensor(camera.FoVx) / 2))
    fy = H / (2 * torch.tan(torch.tensor(camera.FoVy) / 2))
    cx, cy = W / 2.0, H / 2.0

    # 3. 反投影到相机坐标系 (Camera Space)
    z_cam = depth_map.squeeze(0).squeeze(0)  # [H, W]
    x_cam = (u - cx) * z_cam / fx
    y_cam = (v - cy) * z_cam / fy

    # [H, W, 3] -> [H*W, 3]
    xyz_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1).view(-1, 3)

    # 4. 转到世界坐标系 (World Space)
    # 3DGS camera.world_view_transform 是 World -> View 的矩阵 (R|T)
    # 我们需要 View -> World (即 inverse)
    c2w = torch.inverse(camera.world_view_transform.transpose(0, 1))

    # 齐次坐标变换
    xyz_cam_homo = torch.cat([xyz_cam, torch.ones_like(xyz_cam[:, :1])], dim=1)  # [N, 4]

    # [N, 4] @ [4, 4] -> [N, 4]
    xyz_world = (c2w @ xyz_cam_homo.T).T

    return xyz_world[..., :3].view(1, H, W, 3).permute(0, 3, 1, 2)  # [1, 3, H, W]


def reproject_to_view(world_points, target_camera):
    """
    将世界坐标投影到目标视角
    world_points: [1, 3, H, W]
    return:
        grid: [1, H, W, 2] 采样网格 (归一化 -1~1)
        proj_depth: [1, 1, H, W] 投影过去的点的深度 (用于遮挡检测)
    """
    B, C, H, W = world_points.shape
    # [N, 3]
    points_flat = world_points.permute(0, 2, 3, 1).reshape(-1, 3)
    points_homo = torch.cat([points_flat, torch.ones_like(points_flat[:, :1])], dim=1)

    # 1. World -> Target Camera View
    w2c = target_camera.world_view_transform.transpose(0, 1)  # [4, 4]
    cam_points = (w2c @ points_homo.T).T  # [N, 4]

    # 提取相机坐标系下的 XYZ
    x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]

    # 2. 投影到像素坐标 (Projection)
    fx = W / (2 * torch.tan(torch.tensor(target_camera.FoVx) / 2))
    fy = H / (2 * torch.tan(torch.tensor(target_camera.FoVy) / 2))
    cx, cy = W / 2.0, H / 2.0

    # 加上 1e-5 防止除以 0
    u = (x * fx) / (z + 1e-5) + cx
    v = (y * fy) / (z + 1e-5) + cy

    # 3. 归一化到 [-1, 1] 供 grid_sample 使用
    u_norm = 2.0 * u / (W - 1) - 1.0
    v_norm = 2.0 * v / (H - 1) - 1.0

    grid = torch.stack([u_norm, v_norm], dim=-1).view(1, H, W, 2)
    proj_depth = z.view(1, 1, H, W)

    return grid, proj_depth


def get_depth_edge_mask(depth, threshold=0.05):
    """
    [Layer 2] 计算深度图边缘 Mask
    原理：深度梯度大的地方是物体边缘，容易产生 Warping 误差，必须剔除。
    """
    # depth: [B, 1, H, W]

    # 简单的 Sobel 算子或差分计算梯度
    # dy: 垂直梯度, dx: 水平梯度
    dy = torch.abs(depth[..., 1:, :] - depth[..., :-1, :])
    dx = torch.abs(depth[..., :, 1:] - depth[..., :, :-1])

    # Padding 回原尺寸
    dy = F.pad(dy, (0, 0, 0, 1))
    dx = F.pad(dx, (0, 1, 0, 0))

    grad_mag = dy + dx

    # 梯度小于阈值的地方是平坦区域 (Valid)，梯度大的地方是边缘 (Invalid)
    # threshold 需要根据你的深度尺度调整，一般 0.05 - 0.1 左右
    edge_mask = (grad_mag < threshold).float()

    return edge_mask


def warp_consistency_loss(
        src_albedo, src_depth, src_cam,
        tgt_albedo, tgt_depth, tgt_cam,
        save_debug_path=None,
        iteration=0,
        margin_ratio=0.10,  # [策略] 容忍度: 只有比邻居亮 10% 以上才算高光
        patch_size=5  # [策略] Patch大小: 使用 5x5 区域均值对比，抗噪
):
    """
    终极版：基于 Patch 的鲁棒单向光度一致性 Loss
    Robust Unidirectional Photometric Consistency Loss (Patch-based)
    """

    # ==========================================
    # 1. 维度标准化 (Standardize Dimensions)
    # ==========================================
    if src_depth.dim() == 4:
        src_depth_3d = src_depth.squeeze(1)
    else:
        src_depth_3d = src_depth

    if src_depth.dim() == 3:
        src_depth_4d = src_depth.unsqueeze(1)
    else:
        src_depth_4d = src_depth

    if tgt_depth.dim() == 3:
        tgt_depth_4d = tgt_depth.unsqueeze(1)
    else:
        tgt_depth_4d = tgt_depth

    if src_albedo.dim() == 3: src_albedo = src_albedo.unsqueeze(0)
    if tgt_albedo.dim() == 3: tgt_albedo = tgt_albedo.unsqueeze(0)

    # ==========================================
    # 2. 几何变换 Warping
    # ==========================================
    xyz_world = depth_point_to_world(src_depth_3d, src_cam)
    grid, projected_z = reproject_to_view(xyz_world, tgt_cam)

    # 采样 Target 视角的 Albedo 和 Depth
    # padding_mode='zeros' 是默认的，这会导致边界外是黑色，所以下面必须有 mask
    warped_tgt_albedo = F.grid_sample(tgt_albedo, grid, align_corners=True, padding_mode='zeros')
    warped_tgt_depth = F.grid_sample(tgt_depth_4d, grid, align_corners=True, padding_mode='zeros')

    # .detach() 极其重要！把 Target 当作不可变的 Ground Truth
    ref_target = warped_tgt_albedo.detach()

    # ==========================================
    # 3. 构建三层防御 Mask (Physics Validity)
    # ==========================================

    # [Layer 1] 有效网格 Mask (Valid Grid)
    # 剔除采样到了图像边界以外(黑色)的像素。防止"向黑边学习"导致整体变暗。
    valid_grid_mask = (grid.abs().max(dim=-1)[0] < 0.99).float().unsqueeze(1)

    # [Layer 2] 几何遮挡 Mask (Occlusion)
    depth_bias = 0.02
    occ_mask = (projected_z < (warped_tgt_depth + depth_bias)).float()

    # [Layer 3] 边缘 Mask (Edge)
    # 剔除深度跳变剧烈的边缘，防止轮廓模糊
    edge_mask = get_depth_edge_mask(src_depth_4d, threshold=0.08)

    # 合并物理有效性 Mask
    physics_mask = valid_grid_mask * occ_mask * edge_mask

    # ==========================================
    # 4. Patch 级高光判定 (Patch-based Detection)
    # ==========================================

    # 定义均值池化 (眯着眼睛看图，忽略噪点)
    def get_patch_mean(img):
        return F.avg_pool2d(
            img,
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2
        )

    # 计算局部均值
    src_mean = get_patch_mean(src_albedo)
    tgt_mean = get_patch_mean(ref_target)

    # 计算相对比率 Ratio = Src / Target
    eps = 1e-5
    ratio_patch = (src_mean + eps) / (tgt_mean + eps)

    # [核心判定]
    # 只有当 (物理有效) AND (区域亮度比邻居亮 10% 以上)
    # 我们才认定这是"必须去除的高光"
    is_specular_mask = (ratio_patch > (1.0 + margin_ratio)).float()

    # 最终计算 Loss 的 Mask
    final_mask = physics_mask * is_specular_mask

    # ==========================================
    # 5. Loss 计算策略 (Luminance + Chroma)
    # ==========================================
    valid_count = final_mask.sum()

    if valid_count < 10:
        loss = torch.tensor(0.0, device=src_albedo.device)
    else:
        # A. 亮度 Loss (Luminance Penalty)
        # 我们希望 Src 的亮度下降，去接近那个"更暗"的 Tgt
        # 使用 Patch 均值计算 Loss，梯度更平滑
        loss_lum = torch.abs(src_mean.mean(dim=1, keepdim=True) - tgt_mean.mean(dim=1, keepdim=True))

        # B. 色度 Loss (Chromaticity Constraint)
        # 即使亮度要下降，我们要求 RGB 的方向(色相)不能变，防止变成灰色
        src_norm = F.normalize(src_albedo, dim=1, eps=1e-6)
        tgt_norm = F.normalize(ref_target, dim=1, eps=1e-6)
        # Cosine Distance: 1 - cos(theta)
        loss_chroma = 1.0 - (src_norm * tgt_norm).sum(dim=1, keepdim=True)

        # 组合: 主攻亮度下降(1.0)，辅攻色度一致(0.1)
        # 这样网络就会学到: "变暗一点，但别改颜色"
        local_loss = (loss_lum + 0.1 * loss_chroma) * final_mask
        loss = local_loss.sum() / (valid_count + 1e-6)

    # ==========================================
    # 6. Debug 可视化 (六联版 - 修复维度问题)
    # ==========================================
    if save_debug_path is not None:
        os.makedirs(save_debug_path, exist_ok=True)
        with torch.no_grad():
            img_src = src_albedo.detach()
            img_tgt_raw = tgt_albedo.detach()
            img_warp = warped_tgt_albedo.detach()

            # --- 制作 Overlay Mask (Src + 红色高亮) ---
            img_overlay = img_src.clone()

            # 1. 压扁 Mask (解决 9 vs 3 通道报错)
            # 只要 RGB 任意通道判定为高光，该像素即为高光
            if final_mask.shape[1] == 3:
                mask_1ch = final_mask.max(dim=1, keepdim=True)[0]
            else:
                mask_1ch = final_mask

            # 2. 扩展为 3 通道用于显示
            mask_3ch = mask_1ch.repeat(1, 3, 1, 1)

            # 3. 红色高亮
            highlight_color = torch.tensor([1.0, 0.0, 0.0], device=img_src.device).view(1, 3, 1, 1)

            img_overlay = torch.where(
                mask_3ch > 0.5,
                img_src * 0.6 + highlight_color * 0.4,
                img_src
            )

            # --- Mask 通道可视化 ---
            vis_mask = torch.zeros_like(img_src)

            # R: Final Mask (真正计算 Loss 的区域)
            vis_mask[:, 0, :, :] = mask_1ch[:, 0, :, :]

            # G: Edge Mask
            edge_vis = edge_mask if edge_mask.shape[1] == 1 else edge_mask.max(dim=1)[0].unsqueeze(1)
            vis_mask[:, 1, :, :] = edge_vis[:, 0, :, :]

            # B: Occ Mask
            geo_vis = physics_mask if physics_mask.shape[1] == 1 else physics_mask.max(dim=1)[0].unsqueeze(1)
            # 这里为了看物理Mask，我们显示 physics_mask 而不仅仅是 occ
            vis_mask[:, 2, :, :] = geo_vis[:, 0, :, :]

            # --- 误差热力图 ---
            # 显示相对比率差异，越亮表示 Src 比 Tgt 亮得越多
            diff_vis = torch.clamp(ratio_patch - 1.0, 0, 1).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * 5.0

            # --- 六图拼接 ---
            combined = torch.cat([
                img_src,  # 1. 当前视角
                img_tgt_raw,  # 2. 邻居原图
                img_warp,  # 3. 邻居Warp图
                img_overlay,  # 4. 红色高亮惩罚区 (最重要!)
                vis_mask,  # 5. Mask分析
                diff_vis  # 6. 强度分析
            ], dim=3)

            file_name = os.path.join(save_debug_path, f"debug_warp_{iteration:05d}.jpg")
            torchvision.utils.save_image(combined, file_name)

    return loss, final_mask