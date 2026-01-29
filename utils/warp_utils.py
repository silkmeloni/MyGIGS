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
        margin=0.03,  # [Layer 3] 容忍度阈值
        patch_size=3
):
    """
    改进版单向光度一致性 Loss (Robust Unidirectional Consistency)
    包含: 几何遮挡Mask + 边缘Mask + 高光容忍度Mask
    """

    # ==========================================
    # 1. 维度标准化 (保持不变)
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
    # 2. 几何变换 Warping (保持不变)
    # ==========================================
    # 这一步需要你的 depth_point_to_world 和 reproject_to_view 辅助函数
    # 假设它们已经定义在 utils 里
    from utils.warp_utils import depth_point_to_world, reproject_to_view

    xyz_world = depth_point_to_world(src_depth_3d, src_cam)
    grid, projected_z = reproject_to_view(xyz_world, tgt_cam)

    # 采样 Target 视角的 Albedo 和 Depth
    warped_tgt_albedo = F.grid_sample(tgt_albedo, grid, align_corners=True, padding_mode='border')
    warped_tgt_depth = F.grid_sample(tgt_depth_4d, grid, align_corners=True, padding_mode='border')

    # ==========================================
    # 3. 构建三层防御 Mask
    # ==========================================

    # --- [Layer 1] 几何遮挡 Mask (Geometric Occlusion) ---
    # 只有当投影深度 和 实际深度 接近时，才说明没有被遮挡
    depth_bias = 0.05
    occ_mask = (projected_z < (warped_tgt_depth + depth_bias)).float()

    # 边界检查 (视锥体外的点不要)
    bound_mask = (grid.abs().max(dim=-1)[0] < 0.95).float().unsqueeze(1)

    valid_geo_mask = occ_mask * bound_mask

    # --- [Layer 2] 边缘 Mask (Edge/Gradient Mask) ---
    # 剔除 Source 深度图中的剧烈边缘，防止纹理在轮廓处糊掉
    edge_mask = get_depth_edge_mask(src_depth_4d, threshold=0.1)

    # 合并前两层 Mask (物理层面的有效性)
    physics_mask = valid_geo_mask * edge_mask

    # --- [Layer 3] 单向高光判定 & 容忍度 (Unidirectional Logic) ---
    # 核心公式: Diff = Src - Warped_Target
    # 我们只惩罚 Diff > margin 的情况 (显著变亮)

    # .detach() 很重要！我们把 Warped Target 当作"Ground Truth"，不传梯度给它
    ref_target = warped_tgt_albedo.detach()
    diff = src_albedo - ref_target

    # 只有当 Src 比 Tgt 显著亮 (大于 margin) 时，Mask 为 1
    # 这种区域被认为是: "疑似高光区域"
    is_specular_mask = (diff > margin).float()

    # 对 diff 取 Patch 平均，增强鲁棒性 (可选)
    # 如果像素级噪声大，建议加上这个
    if patch_size > 1:
        diff_patched = F.avg_pool2d(diff, kernel_size=patch_size, stride=1, padding=patch_size // 2)
        is_specular_mask = (diff_patched > margin).float()

    # ==========================================
    # 4. 最终 Mask 与 Loss 计算
    # ==========================================

    # 最终参与计算的像素必须满足:
    # 1. 物理上没遮挡 (Layer 1)
    # 2. 不是物体边缘 (Layer 2)
    # 3. 确实是显著变亮的高光 (Layer 3)
    final_mask = physics_mask * is_specular_mask

    # 统计有效像素数
    valid_count = final_mask.sum()

    if valid_count < 10:
        loss = torch.tensor(0.0, device=src_albedo.device)
    else:
        # 只惩罚变亮的部分
        # L1 Loss: abs(diff)
        # 注意: 因为 mask 保证了 diff > 0，所以 abs 其实可以不加，但加上更保险
        loss = (torch.abs(diff) * final_mask).sum() / (valid_count + 1e-6)

    # =========================================================
    # 5. Debug 可视化 (修复维度版)
    # =========================================================
    if save_debug_path is not None:
        os.makedirs(save_debug_path, exist_ok=True)
        with torch.no_grad():
            img_src = src_albedo.detach()
            img_tgt_raw = tgt_albedo.detach()
            img_warp = warped_tgt_albedo.detach()

            # --- [修复点] 制作 Overlay Mask ---
            img_overlay = img_src.clone()

            # 1. 处理 Mask 维度
            # final_mask 目前是 [1, 3, H, W]，我们需要把它压扁成 [1, 1, H, W]
            # 逻辑：只要 RGB 任意通道被判定为高光，该像素就算高光 (取 max)
            if final_mask.shape[1] == 3:
                mask_1ch = final_mask.max(dim=1, keepdim=True)[0]
            else:
                mask_1ch = final_mask

            # 2. 扩展为 3 通道用于图片混合 [1, 3, H, W]
            mask_3ch = mask_1ch.repeat(1, 3, 1, 1)

            # 3. 定义高亮颜色 (纯红)
            highlight_color = torch.tensor([1.0, 0.0, 0.0], device=img_src.device).view(1, 3, 1, 1)

            # 4. 混合
            img_overlay = torch.where(
                mask_3ch > 0.5,
                img_src * 0.6 + highlight_color * 0.4,
                img_src
            )

            # --- Mask 通道可视化 ---
            # 为了可视化，我们也需要把 final_mask 和 edge_mask 统一成单通道展示
            vis_mask = torch.zeros_like(img_src)

            # R通道: Final Mask (压扁后的单通道)
            vis_mask[:, 0, :, :] = mask_1ch[:, 0, :, :]

            # G通道: Edge Mask (如果是3通道也压扁)
            if edge_mask.shape[1] == 3:
                edge_mask_vis = edge_mask.max(dim=1, keepdim=True)[0]
            else:
                edge_mask_vis = edge_mask
            vis_mask[:, 1, :, :] = edge_mask_vis[:, 0, :, :]

            # B通道: Occ Mask
            if valid_geo_mask.shape[1] == 3:
                geo_mask_vis = valid_geo_mask.max(dim=1, keepdim=True)[0]
            else:
                geo_mask_vis = valid_geo_mask
            vis_mask[:, 2, :, :] = geo_mask_vis[:, 0, :, :]

            # --- 误差热力图 ---
            # 同样把 diff 压扁或者取平均显示
            diff_vis = torch.clamp(diff, 0, 1).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * 5.0

            # --- 六图拼接 ---
            combined = torch.cat([
                img_src,
                img_tgt_raw,
                img_warp,
                img_overlay,
                vis_mask,
                diff_vis
            ], dim=3)

            file_name = os.path.join(save_debug_path, f"debug_warp_{iteration:05d}.jpg")
            torchvision.utils.save_image(combined, file_name)

    return loss, final_mask