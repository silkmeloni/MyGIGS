import os
import json
from argparse import ArgumentParser
from typing import Dict, Optional

import imageio.v2 as imageio
import numpy as np
import math
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from diff_gaussian_rasterization import Gaussian_SSR
from tqdm import tqdm
from PIL import Image
from lpips import LPIPS
from typing import Dict, Optional, Union
import kornia

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import viridis_cmap, psnr as get_psnr
from utils.loss_utils import ssim as get_ssim


# ==============================================================================
# 🔥 HDR / EnvMap 加载工具 (从 viewer 移植)
# ==============================================================================
def equirect2cubemap(equi_img, face_res=256):
    """Equirectangular (2D [1, 3, H, W]) -> Cubemap (6 Faces [6, 3, Res, Res])"""
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

    cubemap_faces = F.grid_sample(input_img, grid, mode='bilinear', align_corners=False)
    return cubemap_faces


def load_custom_envmap(path):
    print(f"\n🌌 正在加载外部环境贴图: {path} ...")
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

        faces = equirect2cubemap(img_tensor, face_res=256)
        faces = faces.permute(0, 2, 3, 1)  # [6, 256, 256, 3]
        faces = faces.unsqueeze(0)  # [1, 6, 256, 256, 3]

        cubemap.base.data = faces.contiguous()
        cubemap.build_mips()

        print("✅ 外部环境图转换并加载成功！")
        return cubemap
    except Exception as e:
        print(f"❌ 加载外部环境图失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# 🔥 材质编辑器
# ==============================================================================
def apply_material_editing(gaussians, edit_mode: str):
    """科研级 PBR 材质编辑器"""
    if edit_mode == "none":
        return

    print(f"\n🪄 正在应用材质编辑: [{edit_mode}] ...")
    with torch.no_grad():
        if not hasattr(gaussians, '_albedo'):
            print("⚠️ 警告: 该模型似乎没有 PBR 属性，跳过编辑。")
            return

        albedo = gaussians._albedo
        roughness = gaussians._roughness
        metallic = gaussians._metallic

        if edit_mode == "gold":
            gold_color = torch.tensor([1.0, 0.84, 0.0], device="cuda", dtype=torch.float32)
            albedo.data.copy_(gold_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.15))
            metallic.data.copy_(torch.full_like(metallic, 1.0))
            print("✅ 已切换为: 纯金 (Gold) - 论文常用，展示有色金属高光")

        elif edit_mode == "copper":
            copper_color = torch.tensor([0.95, 0.64, 0.54], device="cuda", dtype=torch.float32)
            albedo.data.copy_(copper_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.2))
            metallic.data.copy_(torch.full_like(metallic, 1.0))
            print("✅ 已切换为: 纯铜 (Copper)")

        elif edit_mode == "silver":
            silver_color = torch.tensor([0.97, 0.97, 0.97], device="cuda", dtype=torch.float32)
            albedo.data.copy_(silver_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.05))
            metallic.data.copy_(torch.full_like(metallic, 1.0))
            print("✅ 已切换为: 镀铬/银 (Silver/Chrome) - 镜面反射测试")

        elif edit_mode == "plastic":
            plastic_color = torch.tensor([0.8, 0.05, 0.05], device="cuda", dtype=torch.float32)
            albedo.data.copy_(plastic_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.15))
            metallic.data.copy_(torch.full_like(metallic, 0.0))
            print("✅ 已切换为: 光泽红塑料 (Shiny Plastic) - 电介质菲涅尔测试")

        elif edit_mode == "rubber":
            rubber_color = torch.tensor([0.1, 0.1, 0.1], device="cuda", dtype=torch.float32)
            albedo.data.copy_(rubber_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.9))
            metallic.data.copy_(torch.full_like(metallic, 0.0))
            print("✅ 已切换为: 哑光橡胶 (Matte Rubber) - 极高粗糙度漫反射测试")

        elif edit_mode == "obsidian":
            obsidian_color = torch.tensor([0.02, 0.02, 0.02], device="cuda", dtype=torch.float32)
            albedo.data.copy_(obsidian_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.02))
            metallic.data.copy_(torch.full_like(metallic, 0.0))
            print("✅ 已切换为: 黑曜石 (Obsidian) - 纯高光轮廓测试")

        elif edit_mode == "clay":
            clay_color = torch.tensor([0.6, 0.35, 0.25], device="cuda", dtype=torch.float32)
            albedo.data.copy_(clay_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.85))
            metallic.data.copy_(torch.full_like(metallic, 0.0))
            print("✅ 已切换为: 黏土 (Clay) - 粗糙表面光影测试")

        elif edit_mode == "wet":
            albedo.data.copy_(albedo.data * 0.7)
            wet_roughness = torch.clamp(roughness.data - 0.4, min=0.05, max=1.0)
            roughness.data.copy_(wet_roughness)
            print("✅ 已切换为: 打湿效果 (Wet)")

        elif edit_mode == "ceramic":
            roughness.data.copy_(torch.full_like(roughness, 0.05))
            metallic.data.copy_(torch.full_like(metallic, 0.0))
            print("✅ 已切换为: 陶瓷材质 (Ceramic)")

        else:
            print(f"⚠️ 未知的编辑模式: {edit_mode}，将不应用任何修改。")


def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
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
        linear0 = 25 / 323 * srgb
        linear1 = ((srgb + 0.055) / 1.055) ** 2.4
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        linear0 = 25 / 323 * srgb
        linear1 = ((srgb + 0.055) / 1.055) ** 2.4
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError


def render_set(
        model_path: str,
        name: str,
        scene: Scene,
        light: CubemapLight,
        pipeline: GroupParams,
        pbr: bool = False,
        metallic: bool = False,
        tone: bool = False,
        gamma: bool = False,
        radius: float = 0.8,
        bias: float = 0.01,
        thick: float = 0.05,
        delta: float = 0.0625,
        step: int = 16,
        start: int = 8,
        indirect: bool = False,
        edit_mode: str = "none",
        envmap_name: str = "ckpt",  # <--- 接收环境图名字
) -> None:
    iteration = scene.loaded_iter
    if name == "train":
        views = scene.getTrainCameras()
    elif name == "test":
        views = scene.getTestCameras()
    else:
        raise ValueError

    light.build_mips()

    # === 🔥 动态输出路径构造 🔥 ===
    env_suffix = f"_env_{envmap_name}" if envmap_name != "ckpt" else ""
    edit_prefix = f"edit_{edit_mode}" if edit_mode != "none" else "ours"
    folder_name = f"{edit_prefix}{env_suffix}_{iteration}"

    os.makedirs(os.path.join(model_path, name), exist_ok=True)

    # 将本次使用的光照图也保存在专属文件夹里，方便核对
    render_dir = os.path.join(model_path, name, folder_name)
    os.makedirs(render_dir, exist_ok=True)

    envmap = light.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    torchvision.utils.save_image(envmap, os.path.join(render_dir, "envmap_used.png"))

    render_path = os.path.join(render_dir, "renders")
    gts_path = os.path.join(render_dir, "gt")
    depths_path = os.path.join(render_dir, "depth")
    normals_path = os.path.join(render_dir, "normal")
    pbr_path = os.path.join(render_dir, "pbr")
    pc_path = os.path.join(render_dir, "pc")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(depths_path, exist_ok=True)
    os.makedirs(normals_path, exist_ok=True)
    os.makedirs(pbr_path, exist_ok=True)
    os.makedirs(pc_path, exist_ok=True)

    brdf_lut = get_brdf_lut().cuda()
    canonical_rays = scene.get_canonical_rays()

    ref_view = views[0]
    H, W = ref_view.image_height, ref_view.image_width
    c2w = torch.inverse(ref_view.world_view_transform.T)  # [4, 4]
    view_dirs_ = (
        (canonical_rays[:, None, :] * c2w[None, :3, :3]).sum(dim=-1).reshape(H, W, 3)
    )
    norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(H, W, 1)

    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0
    lpips_fn = LPIPS(net="vgg").cuda()

    for idx, view in enumerate(tqdm(views, desc=f"Rendering ({folder_name})")):
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        rendering_result = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=background,
            inference=True,
            pad_normal=True,
            derive_normal=True,
            radius=radius,
            bias=bias,
            thick=thick,
            delta=delta,
            step=step,
            start=start
        )

        tanfovx = math.tan(view.FoVx * 0.5)
        tanfovy = math.tan(view.FoVy * 0.5)
        image_height = int(view.image_height)
        image_width = int(view.image_width)

        gt_image = view.original_image[0:3, :, :].cuda()
        alpha_mask = view.gt_alpha_mask.cuda()
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
        depth_map = rendering_result["depth_map"]

        depth_img = viridis_cmap(depth_map.squeeze().cpu().numpy())
        depth_img = (depth_img * 255).astype(np.uint8)
        normal_map_from_depth = rendering_result["normal_map_from_depth"]
        normal_map = rendering_result["normal_map"]
        normal_mask = rendering_result["normal_mask"]

        H, W = view.image_height, view.image_width
        c2w = torch.inverse(view.world_view_transform.T)
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])
            .sum(dim=-1)
            .reshape(H, W, 3)
        )

        occlusion = rendering_result["occlusion_map"].permute(1, 2, 0)

        torchvision.utils.save_image(
            (normal_map_from_depth + 1) / 2,
            os.path.join(normals_path, f"{idx:05d}_from_depth.png"),
        )

        if pbr:
            albedo_map = rendering_result["albedo_map"]
            roughness_map = rendering_result["roughness_map"]
            metallic_map = rendering_result["metallic_map"]
            out_normal_view = rendering_result["out_normal_view"]
            depth_pos = rendering_result["depth_pos"]

            pbr_result = pbr_shading(
                light=light,
                normals=normal_map.permute(1, 2, 0),
                view_dirs=view_dirs,
                mask=normal_mask.permute(1, 2, 0),
                albedo=albedo_map.permute(1, 2, 0),
                roughness=roughness_map.permute(1, 2, 0),
                metallic=metallic_map.permute(1, 2, 0) if metallic else None,
                tone=tone,
                gamma=gamma,
                occlusion=occlusion,
                brdf_lut=brdf_lut,
            )
            render_rgb = pbr_result["render_rgb"].permute(2, 0, 1)

            diffuse_rgb = pbr_result["diffuse_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
            specular_rgb = pbr_result["specular_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)

            render_rgb = torch.where(normal_mask, render_rgb, background[:, None, None])
            diffuse_rgb = torch.where(normal_mask, diffuse_rgb, background[:, None, None])
            specular_rgb = torch.where(normal_mask, specular_rgb, background[:, None, None])

            SSR = Gaussian_SSR(tanfovx, tanfovy, image_width, image_height, radius, bias, thick, delta, step, start)
            if metallic:
                F0 = (1.0 - metallic) * 0.04 + albedo_map * metallic_map
            else:
                F0 = torch.ones_like(albedo_map) * 0.04
                metallic_map = torch.zeros_like(roughness_map)

            linear_rgb = srgb_to_linear(render_rgb)

            (IRR, _) = SSR(out_normal_view, depth_pos, linear_rgb, albedo_map, roughness_map, metallic_map, F0)
            IRR2 = IRR
            IRR2 = linear_to_srgb(IRR2)
            IRR = kornia.filters.median_blur(IRR[None, ...], (3, 3))[0]
            IRR2 = kornia.filters.median_blur(IRR2[None, ...], (3, 3))[0]

            render_rgb = render_rgb + IRR2
            render_rgb = torch.where(normal_mask, render_rgb, background[:, None, None])

            albedo_map = (albedo_map * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
            roughness_map = (roughness_map * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0,
                                                                                                                1.0)
            metallic_map = (metallic_map * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

            bg_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
            normal_map = normal_map * alpha_mask + bg_normal[:, None, None] * (1.0 - alpha_mask)

            diffuse_rgb = (diffuse_rgb * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
            specular_rgb = (specular_rgb * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

            brdf_map = torch.cat([albedo_map, roughness_map, metallic_map], dim=2)
            occlusion = occlusion.permute(2, 0, 1)
            occlusion = (occlusion * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
            to_pil = transforms.ToPILImage()
            occlusion_img = to_pil(occlusion)

            torchvision.utils.save_image((normal_map + 1) / 2, os.path.join(normals_path, f"{idx:05d}_normal.png"))
            occlusion_img.save(os.path.join(pbr_path, f"{idx:05d}occlusion.png"))
            torchvision.utils.save_image(brdf_map, os.path.join(pbr_path, f"{idx:05d}_brdf.png"))
            torchvision.utils.save_image(albedo_map, os.path.join(pbr_path, f"{idx:05d}_albedo.png"))
            torchvision.utils.save_image(roughness_map, os.path.join(pbr_path, f"{idx:05d}_roughness.png"))
            torchvision.utils.save_image(metallic_map, os.path.join(pbr_path, f"{idx:05d}_metallic.png"))

            torchvision.utils.save_image(render_rgb, os.path.join(render_path, f"{idx:05d}.png"))
            torchvision.utils.save_image(render_rgb, os.path.join(pbr_path, f"{idx:05d}.png"))

            torchvision.utils.save_image(diffuse_rgb, os.path.join(pbr_path, f"{idx:05d}_diffuse.png"))
            torchvision.utils.save_image(specular_rgb, os.path.join(pbr_path, f"{idx:05d}_specular.png"))
            torchvision.utils.save_image(render_rgb - IRR2, os.path.join(pbr_path, f"{idx:05d}_DIR.png"))
            torchvision.utils.save_image((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()),
                                         os.path.join(depths_path, f"{idx:05d}_depth.png"))
            torchvision.utils.save_image(IRR2, os.path.join(pbr_path, f"{idx:05d}_indirect.png"))

            if edit_mode == "none" and envmap_name == "ckpt":
                psnr_avg += get_psnr(gt_image, render_rgb).mean().double()
                ssim_avg += get_ssim(gt_image, render_rgb).mean().double()
                lpips_avg += lpips_fn(gt_image, render_rgb).mean().double()

    if pbr and edit_mode == "none" and envmap_name == "ckpt":
        psnr = psnr_avg / len(views)
        ssim = ssim_avg / len(views)
        lpips = lpips_avg / len(views)
        print(f"psnr_avg: {psnr}; ssim_avg: {ssim}; lpips_avg: {lpips}")


@torch.no_grad()
def launch(
        model_path: str,
        checkpoint_path: str,
        dataset: GroupParams,
        pipeline: GroupParams,
        skip_train: bool,
        skip_test: bool,
        pbr: bool = False,
        metallic: bool = False,
        tone: bool = False,
        gamma: bool = False,
        radius: float = 0.8,
        bias: float = 0.01,
        thick: float = 0.05,
        delta: float = 0.0625,
        step: int = 16,
        start: int = 8,
        indirect: bool = False,
        brdf_eval: bool = False,
        edit_mode: str = "none",
        envmap_path: str = None,  # <--- 接收新环境图路径
) -> None:
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    cubemap = CubemapLight(base_res=256).cuda()

    checkpoint = torch.load(checkpoint_path)
    model_params = checkpoint["gaussians"]
    cubemap_params = checkpoint["cubemap"]

    # 1. 恢复几何与材质
    gaussians.restore(model_params)

    # 2. 应用材质编辑
    apply_material_editing(gaussians, edit_mode=edit_mode)

    # 3. 决定加载哪个环境图
    envmap_name = "ckpt"
    if envmap_path and os.path.exists(envmap_path):
        custom_cubemap = load_custom_envmap(envmap_path)
        if custom_cubemap is not None:
            cubemap = custom_cubemap
            # 提取图片名字 (例如 sunset.hdr -> sunset)
            envmap_name = os.path.splitext(os.path.basename(envmap_path))[0]
        else:
            cubemap.load_state_dict(cubemap_params)
    else:
        cubemap.load_state_dict(cubemap_params)

    cubemap.eval()

    if brdf_eval:
        if not skip_train:
            eval_brdf(data_root=dataset.source_path, scene=scene, model_path=model_path, name="train",
                      edit_mode=edit_mode, envmap_name=envmap_name)
        if not skip_test:
            eval_brdf(data_root=dataset.source_path, scene=scene, model_path=model_path, name="test",
                      edit_mode=edit_mode, envmap_name=envmap_name)
    else:
        if not skip_train:
            render_set(
                model_path=model_path, name="train", scene=scene, light=cubemap, pipeline=pipeline,
                pbr=pbr, metallic=metallic, tone=tone, gamma=gamma, radius=radius, bias=bias,
                thick=thick, delta=delta, step=step, start=start, indirect=indirect,
                edit_mode=edit_mode, envmap_name=envmap_name  # <--- 传给渲染组
            )
        if not skip_test:
            render_set(
                model_path=model_path, name="test", scene=scene, light=cubemap, pipeline=pipeline,
                pbr=pbr, metallic=metallic, tone=tone, gamma=gamma, radius=radius, bias=bias,
                thick=thick, delta=delta, step=step, start=start, indirect=indirect,
                edit_mode=edit_mode, envmap_name=envmap_name
            )


def eval_brdf(data_root: str, scene: Scene, model_path: str, name: str, edit_mode: str = "none",
              envmap_name: str = "ckpt") -> None:
    if name == "train":
        transform_file = os.path.join(data_root, "transforms_train.json")
    elif name == "test":
        transform_file = os.path.join(data_root, "transforms_test.json")

    with open(transform_file, "r") as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]

    iteration = scene.loaded_iter

    # 兼容 Eval 路径
    env_suffix = f"_env_{envmap_name}" if envmap_name != "ckpt" else ""
    edit_prefix = f"edit_{edit_mode}" if edit_mode != "none" else "ours"
    folder_name = f"{edit_prefix}{env_suffix}_{iteration}"

    pbr_dir = os.path.join(model_path, name, folder_name, "pbr")
    # ... 下方评估代码保持不变 ...

    # [为了简洁，此处省略原本你 eval_brdf 里的代码，内容一模一样]
    # ...
    pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--pbr", action="store_true",
                        help="Enable pbr rendering for NVS evaluation and export BRDF map.")
    parser.add_argument("--tone", action="store_true", help="Enable aces film tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear_to_sRGB for gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Enable metallic material reconstruction.")
    parser.add_argument("--radius", default=0.8, type=float, help="Path tracing range")
    parser.add_argument("--bias", default=0.01, type=float, help="ensure hit the surface")
    parser.add_argument("--thick", default=0.05, type=float, help="thickness of the surface")
    parser.add_argument("--delta", default=0.0625, type=float, help="angle interval to control the num-sample")
    parser.add_argument("--step", default=16, type=int, help="Path tracing steps")
    parser.add_argument("--start", default=8, type=int, help="Path tracing starting point")
    parser.add_argument("--indirect", action="store_true", help="Enable indirect diffuse modeling.")
    parser.add_argument("--brdf_eval", action="store_true", help="Enable to evaluate reconstructed BRDF.")

    # === 🔥 增加的新参数 🔥 ===
    parser.add_argument("--edit_mode", type=str, default="none",
                        choices=["none", "gold", "copper", "silver", "plastic", "rubber", "obsidian", "clay", "wet",
                                 "ceramic"],
                        help="Material editing mode.")
    parser.add_argument("--envmap", type=str, default=None,
                        help="Path to an external environment map (.hdr, .exr, .png, .jpg) to replace the original one.")

    args = get_combined_args(parser)

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)
    safe_state(args.quiet)

    launch(
        model_path=model_path,
        checkpoint_path=args.checkpoint,
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        pbr=args.pbr,
        metallic=args.metallic,
        tone=args.tone,
        gamma=args.gamma,
        radius=args.radius,
        bias=args.bias,
        thick=args.thick,
        delta=args.delta,
        step=args.step,
        start=args.start,
        indirect=args.indirect,
        brdf_eval=args.brdf_eval,
        edit_mode=args.edit_mode,  # 传入编辑模式
        envmap_path=args.envmap,  # 传入环境图路径
    )