import os
import sys
import traceback

# === 🔥 0. 性能与环境设置 ===
if "CUDA_LAUNCH_BLOCKING" in os.environ:
    del os.environ["CUDA_LAUNCH_BLOCKING"]
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# 自动获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import time
import torch
import torch.nn.functional as F
import viser
import argparse
import math
import numpy as np
import gc
import imageio.v2 as imageio
import threading
from torch import nn
from collections import deque

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, Gaussian_SSR
    from scene.gaussian_model import GaussianModel
    from pbr import CubemapLight, get_brdf_lut, pbr_shading
    from utils.general_utils import safe_state
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


# ==============================================================================
# 1. 全局状态
# ==============================================================================
class RendererState:
    def __init__(self):
        self.gaussians = None
        self.cubemap = None
        self.is_loading = False
        self.lock = threading.Lock()

    def set_models(self, gaussians, cubemap):
        with self.lock:
            self.gaussians = gaussians
            self.cubemap = cubemap
            self.is_loading = False

    def update_cubemap(self, cubemap):
        with self.lock:
            self.cubemap = cubemap

    def update_gaussians(self, gaussians):
        with self.lock:
            self.gaussians = gaussians


state = RendererState()


# ==============================================================================
# 2. 渲染核心 (基于你提供的正确版本)
# ==============================================================================
def fast_render_viewer(
        viewpoint_camera,
        pc: GaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        sh_degree_override: int = -1,
        derive_normal: bool = True,
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
        debug=False,
        inference=True,
        argmax_depth=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 显式 float32
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=torch.float32, requires_grad=False, device="cuda")

    try:
        normal = pc.get_normal
        albedo = pc.get_albedo
        roughness = pc.get_roughness
        metallic = pc.get_metallic
    except AttributeError:
        N = pc.get_xyz.shape[0]
        normal = torch.zeros((N, 3), device="cuda", dtype=torch.float32)
        albedo = torch.ones((N, 3), device="cuda", dtype=torch.float32) * 0.8
        roughness = torch.ones((N, 1), device="cuda", dtype=torch.float32) * 0.5
        metallic = torch.zeros((N, 1), device="cuda", dtype=torch.float32)

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
        shs=pc.get_features,
        colors_precomp=None,
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
        derive_normal=derive_normal
    )

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
        "normal_mask": opacity_map > 0.05,
        "occlusion_map": occlusion_map
    }


def patch_gaussian_model(gaussians):
    print("⚡ 应用显存优化...")
    torch.cuda.empty_cache()
    with torch.no_grad():
        flat_features = torch.cat((gaussians._features_dc, gaussians._features_rest), dim=1).contiguous()
        gaussians._frozen_features = flat_features

        def fast_get_features(self):
            if hasattr(self, "_frozen_features"): return self._frozen_features
            return torch.cat((self._features_dc, self._features_rest), dim=1)

        GaussianModel.get_features = property(fast_get_features)

        for name in ["_xyz", "_opacity", "_scaling", "_rotation", "_normal", "_albedo", "_roughness", "_metallic"]:
            if hasattr(gaussians, name):
                attr = getattr(gaussians, name)
                if isinstance(attr, torch.nn.Parameter): setattr(gaussians, name, attr.data)
                if isinstance(getattr(gaussians, name), torch.Tensor): getattr(gaussians, name).requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()


# ==============================================================================
# 3. 环境图加载 (包含正确的 Equirect2Cube)
# ==============================================================================
def equirect2cubemap(equi_img, face_res=256):
    lin = torch.linspace(-1, 1, face_res, device="cuda")
    u, v = torch.meshgrid(lin, lin, indexing="xy")

    dirs = torch.stack([
        torch.stack((torch.ones_like(u), -v, -u), dim=-1),  # +X
        torch.stack((-torch.ones_like(u), -v, u), dim=-1),  # -X
        torch.stack((u, torch.ones_like(u), v), dim=-1),  # +Y
        torch.stack((u, -torch.ones_like(u), -v), dim=-1),  # -Y
        torch.stack((u, -v, torch.ones_like(u)), dim=-1),  # +Z
        torch.stack((-u, -v, -torch.ones_like(u)), dim=-1)  # -Z
    ])

    dirs = F.normalize(dirs, dim=-1)
    x, y, z = dirs.unbind(-1)

    theta = torch.atan2(x, z)
    phi = torch.asin(y)

    grid_u = theta / math.pi
    grid_v = phi / (math.pi / 2.0)
    grid = torch.stack((grid_u, grid_v), dim=-1)

    if equi_img.dim() == 3: equi_img = equi_img.unsqueeze(0)
    input_img = equi_img.expand(6, -1, -1, -1)

    return F.grid_sample(input_img, grid, mode='bilinear', align_corners=False)


def create_default_cubemap():
    """默认灰色环境 (修复维度 [6, H, W, 3])"""
    print("⚙️ 创建默认灰色环境...")
    cubemap = CubemapLight(base_res=256).cuda()
    # 严格 4D: [Faces, H, W, RGB]
    target_shape = [6, 256, 256, 3]
    grey_data = torch.full(target_shape, 0.15, device="cuda", dtype=torch.float32)
    cubemap.base.data = grey_data
    cubemap.build_mips()
    return cubemap


def load_hdr_cubemap(path):
    print(f"📥 加载环境图: {path}")
    try:
        img_data = imageio.imread(path)
        img_tensor = torch.from_numpy(img_data).float()

        if img_data.dtype == np.uint8:
            img_tensor = img_tensor / 255.0
            if img_tensor.shape[-1] == 4: img_tensor = img_tensor[..., :3]
            img_tensor = srgb_to_linear(img_tensor)
        else:
            if img_tensor.shape[-1] == 4: img_tensor = img_tensor[..., :3]

        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
        cubemap = CubemapLight(base_res=256).cuda()

        print("🔄 正在将全景图转换为 Cubemap...")
        faces = equirect2cubemap(img_tensor, face_res=256)
        faces = faces.permute(0, 2, 3, 1)  # [6, 256, 256, 3]

        cubemap.base.data = faces.contiguous()
        cubemap.build_mips()
        print("✅ 环境图就绪")
        return cubemap
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        traceback.print_exc()
        return None


# ==============================================================================
# 4. Math Helpers
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
        (1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
         two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
         two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def srgb_to_linear(srgb):
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear):
    return torch.where(linear <= 0.0031308, 12.92 * linear, 1.055 * (linear ** (1.0 / 2.4)) - 0.055)


# ==============================================================================
# 5. Main
# ==============================================================================
def main(args):
    print(f"启动 Viser (Port: {args.port})...")
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="collapsible")

    gaussians = GaussianModel(args.sh_degree)
    cubemap = create_default_cubemap()

    init_ply_name = "None"
    init_hdr_name = "Default (Grey)"

    if args.checkpoint:
        print(f"Loading: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        gaussians.restore(checkpoint["gaussians"])
        if "cubemap" in checkpoint:
            # 兼容性处理
            pass
        cubemap.eval()
        patch_gaussian_model(gaussians)
        cubemap.build_mips()
        state.set_models(gaussians, cubemap)
        init_ply_name = args.checkpoint
    elif args.ply_path:
        print(f"Loading: {args.ply_path}")
        gaussians.load_ply(args.ply_path)
        patch_gaussian_model(gaussians)
        state.set_models(gaussians, cubemap)
        init_ply_name = os.path.basename(args.ply_path)
    else:
        state.set_models(None, cubemap)

    brdf_lut = get_brdf_lut().cuda()

    # === GUI ===
    with server.gui.add_folder("📂 Assets Loader"):
        server.gui.add_markdown("### Gaussian Model")
        gui_current_ply = server.gui.add_text("Current PLY", initial_value=init_ply_name, disabled=True)

        default_search = os.path.dirname(args.ply_path) if args.ply_path else CURRENT_DIR
        gui_ply_dir = server.gui.add_text("Search Dir", initial_value=default_search)
        gui_ply_dropdown = server.gui.add_dropdown("Files", options=["(Scan first)"], initial_value="(Scan first)")

        gui_ply_scan = server.gui.add_button("🔄 Scan PLY")
        gui_ply_load = server.gui.add_button("🚀 Load PLY")

        def scan_files(path, exts):
            if not os.path.exists(path): return []
            res = []
            try:
                for f in os.listdir(path):
                    if any(f.lower().endswith(e) for e in exts): res.append(os.path.join(path, f))
            except:
                pass
            return sorted(res)

        @gui_ply_scan.on_click
        def _(_):
            files = scan_files(gui_ply_dir.value, [".ply"])
            if files:
                gui_ply_dropdown.options = files
                gui_ply_dropdown.value = files[0]
            else:
                gui_ply_dropdown.options = ["No .ply found"]
                gui_ply_dropdown.value = "No .ply found"

        @gui_ply_load.on_click
        def _(_):
            path = gui_ply_dropdown.value
            if not os.path.exists(path): return
            state.is_loading = True
            try:
                if state.gaussians: del state.gaussians; torch.cuda.empty_cache()
                new_g = GaussianModel(args.sh_degree)
                new_g.load_ply(path)
                patch_gaussian_model(new_g)
                state.update_gaussians(new_g)
                gui_current_ply.value = os.path.basename(path)
            except Exception as e:
                print(e)
                traceback.print_exc()
            finally:
                state.is_loading = False

        server.gui.add_markdown("---")
        server.gui.add_markdown("### Environment Map")
        gui_current_hdr = server.gui.add_text("Current Env", initial_value=init_hdr_name, disabled=True)
        gui_hdr_dir = server.gui.add_text("Search Dir", initial_value=default_search)
        gui_hdr_dropdown = server.gui.add_dropdown("Files", options=["(Scan first)"], initial_value="(Scan first)")

        gui_hdr_scan = server.gui.add_button("🔄 Scan Env")
        gui_hdr_load = server.gui.add_button("🚀 Load Env")
        gui_hdr_clear = server.gui.add_button("🗑️ Reset Env", color="red")

        @gui_hdr_scan.on_click
        def _(_):
            files = scan_files(gui_hdr_dir.value, [".hdr", ".exr", ".png", ".jpg", ".jpeg"])
            if files:
                gui_hdr_dropdown.options = files
                gui_hdr_dropdown.value = files[0]
            else:
                gui_hdr_dropdown.options = ["No env found"]
                gui_hdr_dropdown.value = "No env found"

        @gui_hdr_load.on_click
        def _(_):
            path = gui_hdr_dropdown.value
            if not os.path.exists(path): return
            state.is_loading = True
            try:
                new_c = load_hdr_cubemap(path)
                if new_c:
                    state.update_cubemap(new_c)
                    gui_current_hdr.value = os.path.basename(path)
            except Exception as e:
                print(e)
                traceback.print_exc()
            finally:
                state.is_loading = False

        @gui_hdr_clear.on_click
        def _(_):
            def_c = create_default_cubemap()
            state.update_cubemap(def_c)
            gui_current_hdr.value = "Default (Grey)"

    with server.gui.add_folder("Stats"):
        gui_fps = server.gui.add_number("FPS", initial_value=0.0, disabled=True)
        gui_time_raster = server.gui.add_number("Raster ms", initial_value=0.0, disabled=True)
        gui_time_shade = server.gui.add_number("Shade ms", initial_value=0.0, disabled=True)
        gui_resolution = server.gui.add_slider("Max Res", min=100, max=1600, step=100, initial_value=800)

    with server.gui.add_folder("Settings"):
        gui_point_cloud_mode = server.gui.add_checkbox("Debug: Point Cloud", initial_value=False)
        gui_mode = server.gui.add_dropdown("Mode",
                                           options=["PBR Final", "Base Color", "Normal", "Roughness", "Metallic",
                                                    "Occlusion"], initial_value="PBR Final")

        with server.gui.add_folder("PBR Params"):
            gui_exposure = server.gui.add_slider("Exposure", min=0.1, max=5.0, step=0.1, initial_value=1.0)
            gui_use_ssr = server.gui.add_checkbox("Enable SSR (Slow)", initial_value=False)
            gui_tone = server.gui.add_checkbox("Tone Mapping", initial_value=True)
            gui_gamma = server.gui.add_checkbox("Gamma Correction", initial_value=True)
            gui_metallic_mode = server.gui.add_checkbox("Use Metallic Map", initial_value=True)
            gui_roughness_mult = server.gui.add_slider("Roughness Mult", 0.0, 2.0, 0.05, 1.0)

    pcd_handle = None

    @gui_point_cloud_mode.on_update
    def _(_):
        nonlocal pcd_handle
        g = state.gaussians
        if not g: return
        if gui_point_cloud_mode.value:
            xyz = g.get_xyz.cpu().numpy()
            colors = (g.get_features[:, 0, :] * 0.282 + 0.5).clamp(0, 1).cpu().numpy()
            pcd_handle = server.scene.add_point_cloud("/debug_pcd", points=xyz, colors=colors, point_size=0.01)
        else:
            if pcd_handle: pcd_handle.remove(); pcd_handle = None

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        fps_history = deque(maxlen=30)

        @client.camera.on_update
        def _(cam: viser.CameraHandle):
            if state.is_loading: return
            g = state.gaussians
            c = state.cubemap
            if not g or not c: return
            if gui_point_cloud_mode.value: return

            # 🔥🔥 全局 no_grad 修复 RuntimeError 🔥🔥
            with torch.no_grad():
                t0 = time.perf_counter()

                # 1. Rasterize
                torch.cuda.synchronize()
                t_r0 = time.perf_counter()

                c2w = torch.eye(4, dtype=torch.float32)
                q = torch.tensor(cam.wxyz).unsqueeze(0)
                R = quaternion_to_matrix(q).squeeze(0)
                c2w[:3, :3] = R
                c2w[:3, 3] = torch.tensor(cam.position)

                W = gui_resolution.value
                H = int(W / cam.aspect)

                viewpoint_cam = MiniCam(c2w=c2w, width=W, height=H, fovy=cam.fov,
                                        fovx=2 * math.atan(math.tan(cam.fov / 2) * cam.aspect))
                bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

                try:
                    render_pkg = fast_render_viewer(viewpoint_cam, g, bg_color, derive_normal=True)
                except Exception:
                    traceback.print_exc()
                    return

                torch.cuda.synchronize()
                t_r1 = time.perf_counter()

                # 2. Shading (AMP float16)
                mode = gui_mode.value
                final_image = render_pkg["render"]

                try:
                    with torch.cuda.amp.autocast(enabled=True):
                        albedo = render_pkg["albedo_map"]
                        normal = render_pkg["normal_map"]
                        roughness = render_pkg["roughness_map"] * gui_roughness_mult.value
                        metallic = render_pkg["metallic_map"] if gui_metallic_mode.value else None
                        opacity = render_pkg["opacity_map"]

                        # View dirs to CUDA
                        view_dirs = F.normalize(c2w[:3, 2], dim=0).view(1, 1, 3).expand(H, W, 3).cuda()

                        if mode == "PBR Final":
                            pbr_out = pbr_shading(
                                light=c,
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

                            # Exposure
                            if gui_exposure.value != 1.0:
                                render_rgb = render_rgb * gui_exposure.value

                            if gui_use_ssr.value:
                                tanfovx = math.tan(cam.fov * 0.5)
                                tanfovy = math.tan(cam.fov * 0.5)
                                SSR = Gaussian_SSR(tanfovx, tanfovy, W, H, 0.8, 0.01, 0.05, 0.0625, 16, 8)
                                linear_rgb = srgb_to_linear(render_rgb)
                                (IRR, _) = SSR(render_pkg["out_normal_view"], render_pkg["depth_pos"], linear_rgb,
                                               albedo, roughness, metallic, torch.ones_like(albedo) * 0.04)
                                render_rgb = render_rgb + linear_to_srgb(IRR)

                            # 🔥 修复：使用正确的 Alpha 混合公式，而不是 torch.where 🔥
                            # torch.where 会导致半透明边缘变成不透明，产生白边/泛白
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

                except Exception:
                    traceback.print_exc()
                    pass

                img_np = final_image.permute(1, 2, 0).clamp(0, 1).float().detach().cpu().numpy()

                torch.cuda.synchronize()
                t_s1 = time.perf_counter()

                client.set_background_image(img_np, format="jpeg")

                dt = t_s1 - t0
                if dt > 0:
                    fps_history.append(1.0 / dt)
                    gui_fps.value = round(sum(fps_history) / len(fps_history), 1)

                gui_time_raster.value = round((t_r1 - t_r0) * 1000, 1)
                gui_time_shade.value = round((t_s1 - t_r1) * 1000, 1)

    while True: time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()
    main(args)