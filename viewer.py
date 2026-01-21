import time
import torch
import torch.nn.functional as F
import viser
import sys
import os
import argparse
import math
import numpy as np
from torch import nn

# ==============================================================================
# TODO: 请确认这里是你的 GI-GS 项目根目录
# ==============================================================================
PROJECT_PATH = "D:/Projects/MyGIGS"  # <--- 请务必根据实际路径修改！
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

try:
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


# ==============================================================================
# 1. 简易 Pipeline 参数类
# ==============================================================================
class ViewerPipeline:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


# ==============================================================================
# 2. 相机模拟类
# ==============================================================================
class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear=0.01, zfar=100.0):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        # 计算 View Matrix (World -> Camera)
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


# ==============================================================================
# 3. 安全 PBR Shader
# ==============================================================================
def safe_pbr_shading(albedo, normal, roughness, metallic, light_pos, view_pos):
    albedo = torch.nan_to_num(albedo, 0.0).clamp(0, 1)
    normal = torch.nan_to_num(normal, 0.0)
    roughness = torch.nan_to_num(roughness, 0.5).clamp(0.05, 1.0)
    metallic = torch.nan_to_num(metallic, 0.0).clamp(0, 1)

    N = F.normalize(normal, dim=0)
    L_dir = F.normalize(light_pos, dim=0).view(3, 1, 1)
    L = L_dir.expand_as(N)

    NdotL = (N * L).sum(0, keepdim=True).clamp(0, 1)
    diffuse = albedo * (1.0 - metallic)

    # 环境光 (模拟天空光)
    up_vec = torch.tensor([0.0, 1.0, 0.0], device="cuda").view(3, 1, 1)
    NdotUp = (N * up_vec).sum(0, keepdim=True) * 0.5 + 0.5
    sky_color = torch.tensor([0.2, 0.3, 0.5], device="cuda").view(3, 1, 1)
    gnd_color = torch.tensor([0.05, 0.05, 0.05], device="cuda").view(3, 1, 1)
    ambient_val = sky_color * NdotUp + gnd_color * (1 - NdotUp)
    ambient = albedo * ambient_val * 0.3

    specular = metallic * NdotL * 0.8

    color = ambient + (diffuse + specular) * NdotL * 1.5
    return color.clamp(0, 1)


# ==============================================================================
# 4. 主程序
# ==============================================================================
def main(args):
    if not os.path.exists(args.ply_path):
        print(f"[Error] 找不到文件: {args.ply_path}")
        return

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="collapsible")

    print(f"Loading model: {args.ply_path}...")
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.ply_path)
    print(f"✅ Loaded {gaussians.get_xyz.shape[0]} points.")

    pipe = ViewerPipeline()

    # --- UI ---
    with server.gui.add_folder("Settings"):
        # 【新增】FPS 显示
        gui_fps = server.gui.add_number("FPS", initial_value=0.0, disabled=True)

        gui_point_cloud_mode = server.gui.add_checkbox("Debug: Show Point Cloud", initial_value=False)

        gui_mode = server.gui.add_dropdown(
            "Render Mode",
            options=["Albedo (Base Color)", "PBR Shading", "Normal", "Opacity"],
            initial_value="Albedo (Base Color)"
        )

        gui_bg = server.gui.add_rgb("Background", (50, 50, 50))

        with server.gui.add_folder("PBR Params"):
            gui_roughness = server.gui.add_slider("Roughness Mult", 0.0, 2.0, 0.05, 1.0)
            gui_metallic = server.gui.add_slider("Metallic Mult", 0.0, 2.0, 0.05, 1.0)
            gui_light = server.add_transform_controls("Light", position=(0, 5, 2), scale=0.5)

    pcd_handle = None

    # --- Debug Point Cloud Logic ---
    @gui_point_cloud_mode.on_update
    def _(_):
        nonlocal pcd_handle
        if gui_point_cloud_mode.value:
            xyz = gaussians.get_xyz.detach().cpu().numpy()
            shs = gaussians.get_features.detach().cpu().numpy()
            colors = shs[:, 0, :] * 0.28209479177387814 + 0.5
            colors = np.clip(colors, 0, 1)
            pcd_handle = server.scene.add_point_cloud("/debug_pcd", points=xyz, colors=colors, point_size=0.01)
        else:
            if pcd_handle is not None:
                pcd_handle.remove()
                pcd_handle = None

    # --- Render Loop ---
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # 【新增】用于计算 FPS 的状态变量
        last_time = time.time()

        @client.camera.on_update
        def _(cam: viser.CameraHandle):
            nonlocal last_time

            if gui_point_cloud_mode.value:
                return

            # 1. 准备矩阵
            c2w = torch.eye(4, dtype=torch.float32)
            q = torch.tensor(cam.wxyz).unsqueeze(0)
            R = quaternion_to_matrix(q).squeeze(0)
            c2w[:3, :3] = R
            c2w[:3, 3] = torch.tensor(cam.position)

            W = 800
            H = int(W / cam.aspect)

            viewpoint_cam = MiniCam(
                c2w=c2w, width=W, height=H, fovy=cam.fov,
                fovx=2 * math.atan(math.tan(cam.fov / 2) * cam.aspect)
            )

            bg_color = torch.tensor(gui_bg.value, dtype=torch.float32, device="cuda") / 255.0

            # 2. Rasterization
            try:
                render_pkg = render(
                    viewpoint_camera=viewpoint_cam,
                    pc=gaussians,
                    pipe=pipe,
                    bg_color=bg_color,
                    inference=True
                )
            except Exception as e:
                return

            # 3. Buffer Extraction
            image_raw = render_pkg["render"]
            try:
                albedo = render_pkg["albedo_map"]
                normal = render_pkg["normal_map"]
                roughness = render_pkg["roughness_map"]
                metallic = render_pkg["metallic_map"]
                opacity = render_pkg["opacity_map"]
            except KeyError:
                img_np = image_raw.permute(1, 2, 0).detach().cpu().numpy()
                client.set_background_image(img_np)
                return

            # 4. Compositing
            mode = gui_mode.value

            if mode == "PBR Shading":
                light_p = torch.tensor(gui_light.position, dtype=torch.float32, device="cuda")
                cam_p = viewpoint_cam.camera_center
                r_mod = roughness * gui_roughness.value
                m_mod = metallic * gui_metallic.value

                shaded = safe_pbr_shading(albedo, normal, r_mod, m_mod, light_p, cam_p)
                final_image = shaded * opacity + bg_color[:, None, None] * (1 - opacity)

            elif mode == "Albedo (Base Color)":
                final_image = albedo * opacity + bg_color[:, None, None] * (1 - opacity)

            elif mode == "Normal":
                final_image = (normal + 1) * 0.5 * opacity + bg_color[:, None, None] * (1 - opacity)

            elif mode == "Opacity":
                final_image = opacity.repeat(3, 1, 1)

            else:
                final_image = image_raw

            img_np = final_image.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
            client.set_background_image(img_np)

            # 【新增】FPS 计算与更新
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 1.0 / dt
                # 为了防止数字跳变太快看不清，可以选择平滑处理，这里直接显示实时值
                gui_fps.value = round(fps, 1)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="Path to .ply file")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    main(args)