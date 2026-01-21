import os

# === 🔥 核心修复：在导入 torch 之前强行干掉性能杀手 ===
if "CUDA_LAUNCH_BLOCKING" in os.environ:
    del os.environ["CUDA_LAUNCH_BLOCKING"]
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import time
import torch
import torch.nn.functional as F
import viser
import sys
import argparse
import math
import numpy as np
import gc
from torch import nn
from collections import deque

# 设置项目路径
PROJECT_PATH = "D:/Projects/MyGIGS"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, Gaussian_SSR
    from scene.gaussian_model import GaussianModel
    from pbr import CubemapLight, get_brdf_lut, pbr_shading
    from utils.general_utils import safe_state
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


# ==============================================================================
# 1. 本地极速 Render (魔改自 gaussian_renderer/__init__.py)
#    - 移除了 Kornia (太慢)
#    - 移除了 retain_grad (爆显存)
#    - 强制 inference=True
# ==============================================================================
def fast_render_viewer(
        viewpoint_camera,
        pc: GaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        sh_degree_override: int = -1,
        derive_normal: bool = True,  # PBR 需要 Normal
):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        radius=0.8,
        bias=0.01,
        thick=0.05,
        delta=0.0625,
        step=16,
        start=8,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree if sh_degree_override < 0 else sh_degree_override,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,  # 🔥 必须关闭
        inference=True,  # 🔥 必须开启
        argmax_depth=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 构造无梯度的 screen points
    screenspace_points = torch.empty(pc.get_xyz.shape[0], 3, dtype=pc.get_xyz.dtype, device="cuda")

    # 属性获取 (兼容处理)
    try:
        normal = pc.get_normal
        albedo = pc.get_albedo
        roughness = pc.get_roughness
        metallic = pc.get_metallic
    except AttributeError:
        # Fallback for non-GI-GS models
        N = pc.get_xyz.shape[0]
        normal = torch.zeros((N, 3), device="cuda")
        albedo = torch.ones((N, 3), device="cuda")
        roughness = torch.zeros((N, 1), device="cuda")
        metallic = torch.zeros((N, 1), device="cuda")

    # 执行光栅化
    (
        rendered_image,
        radii,
        opacity_map,
        depth_map,
        normal_map_from_depth,
        normal_map,
        occlusion_map,
        albedo_map,
        roughness_map,
        metallic_map,
        out_normal_view,
        depth_pos
    ) = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        opacities=pc.get_opacity,
        normal=normal,
        shs=pc.get_features,  # 这里会用到我们的显存补丁
        colors_precomp=None,
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
        derive_normal=derive_normal
    )

    # 简单的 Normal 归一化 (替代 render.py 里的复杂 mask 和 kornia)
    if normal_map is not None:
        normal_map = F.normalize(normal_map, dim=0, p=2)
        out_normal_view = F.normalize(out_normal_view, dim=0, p=2)

    return {
        "render": rendered_image,
        "albedo_map": albedo_map,
        "normal_map": normal_map,
        "roughness_map": roughness_map,
        "metallic_map": metallic_map,
        "opacity_map": opacity_map,
        "out_normal_view": out_normal_view,
        "depth_pos": depth_pos,
        "normal_mask": opacity_map > 0.05,  # 简单生成 Mask
        "occlusion_map": occlusion_map
    }


# ==============================================================================
# 2. 显存优化补丁 (Monkey Patching)
# ==============================================================================
def patch_gaussian_model(gaussians):
    print("⚡ 应用显存优化补丁 (Freeze Model)...")
    torch.cuda.empty_cache()

    with torch.no_grad():
        # 合并 SH 特征
        flat_features = torch.cat((gaussians._features_dc, gaussians._features_rest), dim=1).contiguous()
        gaussians._frozen_features = flat_features

        # 替换属性方法
        def fast_get_features(self):
            if hasattr(self, "_frozen_features"):
                return self._frozen_features
            return torch.cat((self._features_dc, self._features_rest), dim=1)

        GaussianModel.get_features = property(fast_get_features)

        # 释放原始数据
        del gaussians._features_dc
        del gaussians._features_rest
        gaussians._features_dc = None
        gaussians._features_rest = None

        # 关闭梯度
        for name in ["_xyz", "_opacity", "_scaling", "_rotation", "_normal", "_albedo", "_roughness", "_metallic"]:
            if hasattr(gaussians, name):
                attr = getattr(gaussians, name)
                if isinstance(attr, torch.nn.Parameter):
                    setattr(gaussians, name, attr.data)
                if isinstance(getattr(gaussians, name), torch.Tensor):
                    getattr(gaussians, name).requires_grad_(False)

    gc.collect()
    torch.cuda.empty_cache()
    print("✅ 显存优化完成")


# ==============================================================================
# 3. Camera Helper
# ==============================================================================
class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear=0.01, zfar=100.0):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        w2c = torch.inverse(c2w).cuda()
        self.world_view_transform = w2c.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = c2w[:3, 3].cuda()


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def srgb_to_linear(srgb):
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear):
    return torch.where(linear <= 0.0031308, 12.92 * linear, 1.055 * (linear ** (1.0 / 2.4)) - 0.055)


# ==============================================================================
# 4. Main
# ==============================================================================
def main(args):
    print(f"启动 Viser (Port: {args.port})...")
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="collapsible")

    # 1. 加载模型
    gaussians = GaussianModel(args.sh_degree)
    cubemap = CubemapLight(base_res=256).cuda()

    if args.checkpoint:
        print(f"加载 Checkpoint: {args.checkpoint} ...")
        checkpoint = torch.load(args.checkpoint)
        gaussians.restore(checkpoint["gaussians"])
        cubemap.load_state_dict(checkpoint["cubemap"])
        cubemap.eval()
    elif args.ply_path:
        print(f"加载 PLY: {args.ply_path} (注意：无光照信息，将使用默认光照) ...")
        gaussians.load_ply(args.ply_path)
    else:
        print("Error: 请提供 --checkpoint 或 --ply_path")
        return

    print(f"✅ 点数: {gaussians.get_xyz.shape[0]}")
    patch_gaussian_model(gaussians)

    # 预计算 BRDF LUT
    brdf_lut = get_brdf_lut().cuda()
    cubemap.build_mips()

    # UI Settings
    with server.gui.add_folder("Stats"):
        gui_fps = server.gui.add_number("FPS", initial_value=0.0, disabled=True)
        gui_time_raster = server.gui.add_number("Raster (GPU) ms", initial_value=0.0, disabled=True)
        gui_time_shade = server.gui.add_number("Shade (GPU) ms", initial_value=0.0, disabled=True)
        gui_time_net = server.gui.add_number("Net (CPU) ms", initial_value=0.0, disabled=True)
        gui_resolution = server.gui.add_slider("Max Res", min=100, max=1600, step=100, initial_value=800)

    with server.gui.add_folder("Settings"):
        gui_point_cloud_mode = server.gui.add_checkbox("Debug: Point Cloud", initial_value=False)
        gui_mode = server.gui.add_dropdown("Mode",
                                           options=["PBR Final", "Base Color", "Normal", "Roughness", "Metallic",
                                                    "Occlusion"], initial_value="PBR Final")

        with server.gui.add_folder("PBR Params"):
            gui_use_ssr = server.gui.add_checkbox("Enable SSR (Slow)", initial_value=False)
            gui_tone = server.gui.add_checkbox("Tone Mapping", initial_value=True)
            gui_gamma = server.gui.add_checkbox("Gamma Correction", initial_value=True)
            gui_metallic_mode = server.gui.add_checkbox("Use Metallic Map", initial_value=True)
            gui_roughness_mult = server.gui.add_slider("Roughness Mult", 0.0, 2.0, 0.05, 1.0)

    pcd_handle = None

    @gui_point_cloud_mode.on_update
    def _(_):
        nonlocal pcd_handle
        if gui_point_cloud_mode.value:
            xyz = gaussians.get_xyz.cpu().numpy()
            shs = gaussians.get_features.cpu().numpy()
            colors = shs[:, 0, :] * 0.282 + 0.5
            colors = np.clip(colors, 0, 1)
            pcd_handle = server.scene.add_point_cloud("/debug_pcd", points=xyz, colors=colors, point_size=0.01)
        else:
            if pcd_handle is not None:
                pcd_handle.remove()
                pcd_handle = None

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        fps_history = deque(maxlen=30)

        @client.camera.on_update
        def _(cam: viser.CameraHandle):
            if gui_point_cloud_mode.value: return

            # [Profiling T0]
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Camera Setup
            c2w = torch.eye(4, dtype=torch.float32)
            q = torch.tensor(cam.wxyz).unsqueeze(0)
            R = quaternion_to_matrix(q).squeeze(0)
            c2w[:3, :3] = R
            c2w[:3, 3] = torch.tensor(cam.position)

            target_width = gui_resolution.value
            W = target_width
            H = int(W / cam.aspect)

            viewpoint_cam = MiniCam(c2w=c2w, width=W, height=H, fovy=cam.fov,
                                    fovx=2 * math.atan(math.tan(cam.fov / 2) * cam.aspect))
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

            # === 1. Rasterization ===
            try:
                # 显式传递 derive_normal=True 以支持 PBR
                render_pkg = fast_render_viewer(
                    viewpoint_camera=viewpoint_cam,
                    pc=gaussians,
                    bg_color=bg_color,
                    derive_normal=True
                )
            except Exception as e:
                print(f"Render Error: {e}")
                return

            # [Profiling T1]
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # === 2. PBR Shading Pipeline (Copy from render.py) ===
            mode = gui_mode.value
            final_image = render_pkg["render"]  # Fallback

            try:
                albedo = render_pkg["albedo_map"]
                normal = render_pkg["normal_map"]
                roughness = render_pkg["roughness_map"] * gui_roughness_mult.value
                metallic = render_pkg["metallic_map"] if gui_metallic_mode.value else None
                opacity = render_pkg["opacity_map"]

                # 准备 view_dirs
                # MiniCam 的 view direction 很简单，就是相机中心指向像素
                # 但为了严谨，我们构建 ray direction
                # 这里简化：在 View Space 下，View Dir 就是 (0,0,1) 或者根据像素坐标算
                # render.py 中使用了 canonical_rays，这里为了速度，我们用 View Position - World Position
                # 实际上 pbr_shading 函数里 view_dirs 需要是 [H,W,3]

                # 简易 View Dir 计算 (假设平行投影近似，或者简单的中心辐射)
                # 更精确的做法需要反投影，这里为了 viewer 流畅度做近似
                # view_dirs = (CameraPos - PixelPos).normalize()
                # 鉴于 fast_render 没有返回 depth_pos 的准确世界坐标，我们用相机朝向近似
                view_dir_vec = F.normalize(c2w[:3, 2], dim=0)  # Z axis
                view_dirs = view_dir_vec.view(1, 1, 3).expand(H, W, 3)

                if mode == "PBR Final":
                    # PBR Shading
                    pbr_out = pbr_shading(
                        light=cubemap,
                        normals=normal.permute(1, 2, 0),
                        view_dirs=view_dirs,
                        mask=render_pkg["normal_mask"].permute(1, 2, 0),
                        albedo=albedo.permute(1, 2, 0),
                        roughness=roughness.permute(1, 2, 0),
                        metallic=metallic.permute(1, 2, 0) if metallic is not None else None,
                        tone=gui_tone.value,
                        gamma=gui_gamma.value,
                        occlusion=render_pkg["occlusion_map"].permute(1, 2, 0),
                        brdf_lut=brdf_lut
                    )
                    render_rgb = pbr_out["render_rgb"].permute(2, 0, 1)

                    # SSR (Screen Space Reflections)
                    if gui_use_ssr.value:
                        # 准备 SSR 需要的参数
                        tanfovx = math.tan(cam.fov * 0.5)
                        tanfovy = math.tan(cam.fov * 0.5)  # 近似
                        SSR = Gaussian_SSR(tanfovx, tanfovy, W, H,
                                           0.8, 0.01, 0.05, 0.0625, 16, 8)  # 使用默认参数

                        # SSR 输入准备
                        linear_rgb = srgb_to_linear(render_rgb)
                        # 注意：SSR 需要 [C, H, W]
                        (IRR, _) = SSR(
                            render_pkg["out_normal_view"],
                            render_pkg["depth_pos"],
                            linear_rgb,
                            albedo,
                            roughness,
                            metallic if metallic is not None else torch.zeros_like(roughness),
                            torch.ones_like(albedo) * 0.04  # F0
                        )
                        # SSR 结果叠加
                        render_rgb = render_rgb + linear_to_srgb(IRR)

                    final_image = render_rgb * opacity + bg_color[:, None, None] * (1 - opacity)

                elif mode == "Base Color":
                    final_image = albedo
                elif mode == "Normal":
                    final_image = (normal + 1) * 0.5
                elif mode == "Roughness":
                    final_image = roughness.repeat(3, 1, 1)
                elif mode == "Metallic":
                    final_image = metallic.repeat(3, 1, 1) if metallic is not None else torch.zeros_like(albedo)
                elif mode == "Occlusion":
                    final_image = render_pkg["occlusion_map"].repeat(3, 1, 1)

            except Exception as e:
                # print(f"Shading Error: {e}")
                pass

            img_np = final_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

            # [Profiling T2]
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            client.set_background_image(img_np, format="jpeg")

            # [Profiling T3]
            t3 = time.perf_counter()

            # Update Stats
            dt = t3 - t0
            if dt > 0:
                fps_history.append(1.0 / dt)
                gui_fps.value = round(sum(fps_history) / len(fps_history), 1)

            gui_time_raster.value = round((t1 - t0) * 1000, 1)
            gui_time_shade.value = round((t2 - t1) * 1000, 1)
            gui_time_net.value = round((t3 - t2) * 1000, 1)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()
    main(args)