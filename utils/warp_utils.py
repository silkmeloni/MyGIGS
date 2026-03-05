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


def get_normal_facing_mask(xyz_world, src_normal, tgt_cam, threshold=0.05):
    """
    计算法线朝向掩码，剔除在 Target 视角下背向相机的点。
    xyz_world: [B, 3, H, W] 世界坐标系下的 3D 点
    src_normal: [B, 3, H, W] Source 视角渲染出的世界坐标系法线
    tgt_cam: Target 视角的 3DGS Camera 对象
    threshold: 阈值，大于此值才认为是有效面向相机 (0.05 可过滤极端的边缘侧切面)
    """
    if src_normal is None:
        # 如果未传入法线，返回全 1 的 fallback mask
        return torch.ones((xyz_world.shape[0], 1, xyz_world.shape[2], xyz_world.shape[3]), device=xyz_world.device)

    # 确保 normal 是 4D
    if src_normal.dim() == 3:
        src_normal = src_normal.unsqueeze(0)

    # 1. 获取 Target 相机在世界坐标系下的中心位置
    # 3DGS 的 camera 对象通常自带 camera_center 属性
    if hasattr(tgt_cam, 'camera_center'):
        cam_pos = tgt_cam.camera_center.view(1, 3, 1, 1).to(xyz_world.device)
    else:
        # Fallback: 通过 w2c 矩阵求逆得到相机世界坐标
        w2c = tgt_cam.world_view_transform.transpose(0, 1)  # [4, 4]
        c2w = torch.inverse(w2c)
        cam_pos = c2w[:3, 3].view(1, 3, 1, 1).to(xyz_world.device)

    # 2. 计算从表面点指向 Target 相机的视线方向 (View Direction)
    view_dir = cam_pos - xyz_world
    view_dir = F.normalize(view_dir, dim=1)  # [B, 3, H, W]

    # 3. 确保法线被归一化
    normal_norm = F.normalize(src_normal, dim=1)

    # 4. 计算点乘 (Dot Product)
    dot_prod = (normal_norm * view_dir).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    # 5. 大于 threshold 表示面向相机
    facing_mask = (dot_prod > threshold).float()

    return facing_mask

def material_consistency_loss(
        src_maps, src_depth, src_cam,
        tgt_maps, tgt_depth, tgt_cam,
        src_normal=None,         # [新增] Source 视角的法线图
        constraint_albedo=False,
        patch_size=7,
        save_debug_path=None,
        iteration=0
):
    """
    MaterialRefGS 风格的多视角材质一致性约束 (带 Debug 可视化)
    """

    # 0. 辅助函数：确保 4D [N, C, H, W]
    def ensure_4d(tensor):
        if tensor.dim() == 3: return tensor.unsqueeze(1)
        return tensor

    # 1. 几何 Warping 准备
    if src_depth.dim() == 4: src_depth = src_depth.squeeze(1)
    if tgt_depth.dim() == 3: tgt_depth = tgt_depth.unsqueeze(1)

    xyz_world = depth_point_to_world(src_depth, src_cam)
    grid, projected_z = reproject_to_view(xyz_world, tgt_cam)
    valid_grid_mask = (grid.abs().max(dim=-1)[0] < 0.99).float().unsqueeze(1)

    # 2. 采样 Target
    def warp_map(img):
        img_4d = ensure_4d(img)
        return F.grid_sample(img_4d, grid, align_corners=True, padding_mode='zeros')

    # 键名适配
    r_key = 'roughness_map' if 'roughness_map' in tgt_maps else 'roughness'
    m_key = 'metallic_map' if 'metallic_map' in tgt_maps else 'metallic'
    a_key = 'albedo_map' if 'albedo_map' in tgt_maps else 'albedo'

    src_rough = ensure_4d(src_maps[r_key])
    src_metal = ensure_4d(src_maps[m_key])

    warped_roughness = warp_map(tgt_maps[r_key])
    warped_metallic = warp_map(tgt_maps[m_key])
    warped_tgt_depth = F.grid_sample(tgt_depth, grid, align_corners=True, padding_mode='zeros')

    # 3. Mask
    depth_bias = 0.05
    occ_mask = (projected_z < (warped_tgt_depth + depth_bias)).float()
    # [新增] 计算法线朝向掩码
    facing_mask = get_normal_facing_mask(xyz_world, src_normal, tgt_cam, threshold=0.05)

    # 最终掩码 = 在屏幕内 * 未被遮挡 * 且面向 Target 相机
    valid_mask = valid_grid_mask * occ_mask * facing_mask

    if valid_mask.sum() < 10:
        return torch.tensor(0.0, device=src_depth.device)

    # 4. 计算 Loss
    loss = 0.0

    # Roughness
    loss += F.mse_loss(src_rough * valid_mask, warped_roughness.detach() * valid_mask, reduction='sum') / (
                valid_mask.sum() + 1e-6)
    # Metallic
    loss += F.mse_loss(src_metal * valid_mask, warped_metallic.detach() * valid_mask, reduction='sum') / (
                valid_mask.sum() + 1e-6)

    # Albedo (Optional)
    src_albedo = None
    warped_albedo = None
    if constraint_albedo:
        src_albedo = ensure_4d(src_maps[a_key])
        warped_albedo = warp_map(tgt_maps[a_key])
        loss += F.mse_loss(src_albedo * valid_mask, warped_albedo.detach() * valid_mask, reduction='sum') / (
                    valid_mask.sum() + 1e-6)

    # ==========================================
    # 5. Debug 可视化 (新增模块)
    # ==========================================
    if save_debug_path is not None:
        os.makedirs(save_debug_path, exist_ok=True)
        with torch.no_grad():
            # 辅助函数：转为3通道以便拼接
            def to_3ch(img):
                if img is None: return None
                img = img.detach()
                if img.shape[1] == 1: return img.repeat(1, 3, 1, 1)
                return img

            # 准备 Mask 可视化 (白色为有效区域)
            vis_mask = to_3ch(valid_mask)

            # --- 第一行: Roughness 分析 ---
            row_rough = torch.cat([
                to_3ch(src_rough),  # 1. Src Roughness
                to_3ch(warped_roughness),  # 2. Warped Tgt Roughness (应与1一致)
                vis_mask,  # 3. 有效 Mask (白色为计算区域)
                to_3ch((src_rough - warped_roughness).abs() * valid_mask * 5.0)  # 4. 误差热力 (放大5倍)
            ], dim=3)

            # --- 第二行: Metallic 分析 ---
            row_metal = torch.cat([
                to_3ch(src_metal),
                to_3ch(warped_metallic),
                vis_mask,
                to_3ch((src_metal - warped_metallic).abs() * valid_mask * 5.0)
            ], dim=3)

            # --- 组合所有行 ---
            final_grid = torch.cat([row_rough, row_metal], dim=2)  # 垂直拼接

            # --- 第三行: Albedo 分析 (如果开启) ---
            if constraint_albedo and src_albedo is not None:
                row_albedo = torch.cat([
                    to_3ch(src_albedo),
                    to_3ch(warped_albedo),
                    vis_mask,
                    to_3ch((src_albedo - warped_albedo).abs() * valid_mask * 5.0)
                ], dim=3)
                final_grid = torch.cat([final_grid, row_albedo], dim=2)

            # 保存
            file_name = os.path.join(save_debug_path, f"debug_mat_consist_{iteration:05d}.jpg")
            torchvision.utils.save_image(final_grid, file_name)

    return loss

def warp_consistency_loss(
        src_albedo, src_depth, src_cam,
        tgt_albedo, tgt_depth, tgt_cam,
        src_normal=None,         # [新增] Source 视角的法线图
        save_debug_path=None,
        iteration=0,
        margin=0.03,
        patch_size=7
):
    """
    [能量守恒修正版] 零均值单向一致性 Loss
    Zero-Mean Unidirectional Consistency (Energy Preserving)
    """

    # --- 1. 维度处理 & 2. Warping (保持不变) ---
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

    from utils.warp_utils import depth_point_to_world, reproject_to_view
    xyz_world = depth_point_to_world(src_depth_3d, src_cam)
    grid, projected_z = reproject_to_view(xyz_world, tgt_cam)

    warped_tgt_albedo = F.grid_sample(tgt_albedo, grid, align_corners=True, padding_mode='zeros')
    warped_tgt_depth = F.grid_sample(tgt_depth_4d, grid, align_corners=True, padding_mode='zeros')
    ref_target = warped_tgt_albedo.detach()

    # --- 3. 物理 Mask (保持不变) ---
    valid_grid_mask = (grid.abs().max(dim=-1)[0] < 0.99).float().unsqueeze(1)
    depth_bias = 0.02
    occ_mask = (projected_z < (warped_tgt_depth + depth_bias)).float()

    # [边缘剔除]
    dy = torch.abs(src_depth_4d[..., 1:, :] - src_depth_4d[..., :-1, :])
    dx = torch.abs(src_depth_4d[..., :, 1:] - src_depth_4d[..., :, :-1])
    dy = F.pad(dy, (0, 0, 0, 1));
    dx = F.pad(dx, (0, 1, 0, 0))
    edge_mask = ((dy + dx) < 0.08).float()

    # [新增] 计算法线朝向掩码
    facing_mask = get_normal_facing_mask(xyz_world, src_normal, tgt_cam, threshold=0.05)

    # 乘以 facing_mask 剔除背向面
    physics_mask = valid_grid_mask * occ_mask * edge_mask * facing_mask

    # ==========================================
    # [核心修正] 零均值 (Zero-Mean) 分解
    # ==========================================

    # 1. 计算局部能量 (DC分量)
    # 使用较大的 Patch (如 7x7) 来统计局部照明条件
    def get_local_mean(img):
        return F.avg_pool2d(img, kernel_size=patch_size, stride=1, padding=patch_size // 2)

    src_dc = get_local_mean(src_albedo)
    tgt_dc = get_local_mean(ref_target)

    # 2. 剥离能量，获得纯结构 (AC分量)
    # Src_AC: 这里的正值代表"比周围亮"，负值代表"比周围暗"
    src_ac = src_albedo - src_dc
    tgt_ac = ref_target - tgt_dc

    # ==========================================
    # [单向约束] 只砍高频尖峰
    # ==========================================

    # 逻辑：
    # Src_AC > 0.1 表示 Src 这里有个很亮的尖峰 (相对于它自己的背景)
    # Tgt_AC < 0.0 表示 Tgt 这里很平坦，或者比背景暗
    # 差异 diff > 0 表示 Src 的尖峰程度 远大于 Tgt

    # 我们只比较 AC 分量！这样完全忽略了 DC (整体亮度) 的差异
    diff_ac = src_ac - tgt_ac

    # 判定高光：只有当"结构上的突起"显著大于邻居时
    is_specular_mask = (diff_ac > margin).float()

    final_mask = physics_mask * is_specular_mask
    valid_count = final_mask.sum()

    if valid_count < 10:
        loss = torch.tensor(0.0, device=src_albedo.device)
    else:
        # [Loss 计算]
        # 即使被判定为高光，我们惩罚什么？
        # 我们惩罚的是 diff_ac (结构的差异)，而不是 src_albedo (绝对亮度)
        # 这意味着：梯度只会去削平那个尖峰，而不会把整个地基往下压
        loss = (torch.abs(diff_ac) * final_mask).sum() / (valid_count + 1e-6)

        # 辅以色度保护 (能量守恒的最后一道防线：颜色方向不变)
        src_norm = F.normalize(src_albedo, dim=1, eps=1e-6)
        tgt_norm = F.normalize(ref_target, dim=1, eps=1e-6)
        loss_chroma = 1.0 - (src_norm * tgt_norm).sum(dim=1, keepdim=True)

        # 最终 Loss
        loss += 0.1 * (loss_chroma * final_mask).sum() / (valid_count + 1e-6)

    # ==========================================
    # 6. Debug 可视化
    # ==========================================
    if save_debug_path is not None:
        os.makedirs(save_debug_path, exist_ok=True)
        with torch.no_grad():
            img_src = src_albedo.detach()
            # 可视化 AC 分量 (加0.5变灰度，方便观察)
            img_src_ac = torch.clamp(src_ac + 0.5, 0, 1).detach()
            img_tgt_ac = torch.clamp(tgt_ac + 0.5, 0, 1).detach()

            # 红色高亮 Overlay
            img_overlay = img_src.clone()
            if final_mask.shape[1] == 3:
                mask_1ch = final_mask.max(dim=1, keepdim=True)[0]
            else:
                mask_1ch = final_mask
            mask_3ch = mask_1ch.repeat(1, 3, 1, 1)
            highlight = torch.tensor([1.0, 0.0, 0.0], device=img_src.device).view(1, 3, 1, 1)
            img_overlay = torch.where(mask_3ch > 0.5, img_src * 0.6 + highlight * 0.4, img_src)

            # [Src] [Src_AC(结构)] [Tgt_AC(结构)] [Overlay]
            combined = torch.cat([img_src, img_src_ac, img_tgt_ac, img_overlay], dim=3)
            torchvision.utils.save_image(combined, os.path.join(save_debug_path, f"debug_warp_{iteration:05d}.jpg"))

    return loss, final_mask