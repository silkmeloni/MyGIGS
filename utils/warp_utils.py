# utils/warp_utils.py
import torch
import torch.nn.functional as F
import torchvision # [新增] 用于保存图片
import os          # [新增]


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
    depth_map: [1, H, W]
    camera: 3DGS 的 Camera 对象
    return: [B, 3, H, W] 世界坐标图
    """
    H, W = depth_map.shape[1], depth_map.shape[2]
    device = depth_map.device

    # 1. 获取像素坐标
    grid = get_grid(H, W, device)  # [H, W, 2]
    u, v = grid[..., 0], grid[..., 1]

    # 2. 获取相机内参 (假设 principal point 在中心, focal length 已知)
    # 注意：3DGS 的 camera 对象通常有 fovx, fovy
    # fx = W / (2 * tan(fovx / 2))
    fx = W / (2 * torch.tan(torch.tensor(camera.FoVx) / 2))
    fy = H / (2 * torch.tan(torch.tensor(camera.FoVy) / 2))
    cx, cy = W / 2.0, H / 2.0

    # 3. 反投影到相机坐标系 (Camera Space)
    # Z = depth
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    z_cam = depth_map.squeeze(0)
    x_cam = (u - cx) * z_cam / fx
    y_cam = (v - cy) * z_cam / fy

    # [H, W, 3] -> [H*W, 3]
    xyz_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1).view(-1, 3)

    # 4. 转到世界坐标系 (World Space)
    # 3DGS camera.world_view_transform 是 World -> View 的矩阵 (R|T)
    # 我们需要 View -> World (即 inverse)
    # 注意：3DGS 的矩阵通常是转置存储的，或者是 row-major，这里需小心
    # world_view_transform: [4, 4]
    c2w = torch.inverse(camera.world_view_transform.transpose(0, 1))

    # 齐次坐标变换
    xyz_cam_homo = torch.cat([xyz_cam, torch.ones_like(xyz_cam[:, :1])], dim=1)  # [N, 4]
    xyz_world = (c2w @ xyz_cam_homo.T).T  # [N, 4]

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
    # 使用 target_camera 的投影矩阵 full_proj_transform (World -> NDC)
    # 或者手动算: u = fx * x / z + cx
    # 这里手动算更可控
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


def warp_consistency_loss(
        src_albedo, src_depth, src_cam,
        tgt_albedo, tgt_depth, tgt_cam,
        save_debug_path=None,  # [新增参数] 传入保存路径
        iteration=0  # [新增参数] 传入当前迭代次数
):
    """
    计算重投影一致性 Loss，并支持 Debug 可视化
    """
    # 1. 坐标变换
    xyz_world = depth_point_to_world(src_depth, src_cam)
    grid, projected_z = reproject_to_view(xyz_world, tgt_cam)

    # 2. 采样 Target 图像 (模拟 Source 视角应该看到的样子)
    # 输入必须是 [B, C, H, W]
    if tgt_albedo.dim() == 3: tgt_albedo = tgt_albedo.unsqueeze(0)
    if tgt_depth.dim() == 3: tgt_depth = tgt_depth.unsqueeze(0)

    warped_tgt_albedo = F.grid_sample(tgt_albedo, grid, align_corners=True)  # [1, 3, H, W]
    warped_tgt_depth = F.grid_sample(tgt_depth, grid, align_corners=True)  # [1, 1, H, W]

    # 3. 遮挡剔除
    depth_bias = 0.05
    # projected_z: Source 点投到 Target 下的深度 (计算值)
    # warped_tgt_depth: Target 实际看到的深度 (真实值)
    # 如果 计算值 > 真实值，说明被挡住了
    valid_mask = (projected_z < (warped_tgt_depth + depth_bias)).float()

    # 边界剔除
    mask_bound = (grid.abs().max(dim=-1)[0] < 1.0).float().view(1, 1, src_depth.shape[1], src_depth.shape[2])
    final_mask = valid_mask * mask_bound  # [1, 1, H, W]

    # 4. 计算 Loss
    src_albedo_batch = src_albedo.unsqueeze(0) if src_albedo.dim() == 3 else src_albedo
    diff = torch.abs(src_albedo_batch - warped_tgt_albedo)
    loss = (diff * final_mask).sum() / (final_mask.sum() + 1e-6)

    # ================= [Debug 核心部分] =================
    if save_debug_path is not None:
        os.makedirs(save_debug_path, exist_ok=True)
        with torch.no_grad():
            # 准备拼接图片，为了方便观察，我们把 Mask 变成 3 通道红色覆盖
            # 1. Source View (基准)
            img_src = src_albedo_batch.detach()

            # 2. Warped Target (应该长得像 Source)
            img_warp = warped_tgt_albedo.detach()

            # 3. Target View (原始目标视角，用于参考)
            # 把它 resize 到和 src 一样大以便拼接 (防止分辨率不一致)
            img_tgt_ref = F.interpolate(tgt_albedo, size=img_src.shape[2:], mode='bilinear')

            # 4. Mask (可视化有哪些区域参与了 Loss 计算)
            # 黑白图：白=有效，黑=无效
            img_mask = final_mask.repeat(1, 3, 1, 1).detach()

            # 5. Error Map (差异图，越亮差异越大)
            # 乘以 5 倍亮度方便观察
            img_error = (diff * final_mask).detach() * 5.0

            # 拼成一行: Source | Warped | Mask | Error | Target(Ref)
            combined = torch.cat([img_src, img_warp, img_mask, img_error, img_tgt_ref], dim=3)

            # 保存
            file_name = os.path.join(save_debug_path, f"debug_warp_{iteration:05d}.jpg")
            torchvision.utils.save_image(combined, file_name)
            print(f"[Debug] Saved warp visualization to {file_name}")
    # ===================================================

    return loss, final_mask