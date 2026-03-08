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

        # ==========================================
        # 🧪 第一组：金属测试 (有色高光，高反射)
        # ==========================================
        if edit_mode == "gold":
            gold_color = torch.tensor([1.0, 0.84, 0.0], device="cuda", dtype=torch.float32)
            albedo.data.copy_(gold_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.15))
            metallic.data.copy_(torch.full_like(metallic, 1.0))
            print("✅ 已切换为: 纯金 (Gold) - 论文常用，展示有色金属高光")

        elif edit_mode == "copper":
            # 纯铜：独特的红橙色金属
            copper_color = torch.tensor([0.95, 0.64, 0.54], device="cuda", dtype=torch.float32)
            albedo.data.copy_(copper_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.2))
            metallic.data.copy_(torch.full_like(metallic, 1.0))
            print("✅ 已切换为: 纯铜 (Copper)")

        elif edit_mode == "silver":
            # 镀铬/白银：完美的镜面反射，用于展示 HDR 环境图的清晰度
            silver_color = torch.tensor([0.97, 0.97, 0.97], device="cuda", dtype=torch.float32)
            albedo.data.copy_(silver_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.05))
            metallic.data.copy_(torch.full_like(metallic, 1.0))
            print("✅ 已切换为: 镀铬/银 (Silver/Chrome) - 镜面反射测试")

        # ==========================================
        # 🧪 第二组：电介质/非金属测试 (白色高光，漫反射为主)
        # ==========================================
        elif edit_mode == "plastic":
            # 红色光泽塑料：展示电介质的 Fresnel 现象（基础色是红色，但高光反射是环境光的原本颜色/白色）
            plastic_color = torch.tensor([0.8, 0.05, 0.05], device="cuda", dtype=torch.float32)
            albedo.data.copy_(plastic_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.15))
            metallic.data.copy_(torch.full_like(metallic, 0.0))
            print("✅ 已切换为: 光泽红塑料 (Shiny Plastic) - 电介质菲涅尔测试")

        elif edit_mode == "rubber":
            # 哑光黑橡胶：极高粗糙度，展示低频的漫反射光照，验证光照积分计算的正确性
            rubber_color = torch.tensor([0.1, 0.1, 0.1], device="cuda", dtype=torch.float32)
            albedo.data.copy_(rubber_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.9))
            metallic.data.copy_(torch.full_like(metallic, 0.0))
            print("✅ 已切换为: 哑光橡胶 (Matte Rubber) - 极高粗糙度漫反射测试")

        elif edit_mode == "obsidian":
            # 黑曜石/黑玻璃：Albedo接近0，此时物体完全靠Specular(高光)体现轮廓，验证能量守恒
            obsidian_color = torch.tensor([0.02, 0.02, 0.02], device="cuda", dtype=torch.float32)
            albedo.data.copy_(obsidian_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.02))
            metallic.data.copy_(torch.full_like(metallic, 0.0))
            print("✅ 已切换为: 黑曜石 (Obsidian) - 纯高光轮廓测试")

        elif edit_mode == "clay":
            # 黏土/陶土：典型的粗糙非金属，常用于展示几何法线细节
            clay_color = torch.tensor([0.6, 0.35, 0.25], device="cuda", dtype=torch.float32)
            albedo.data.copy_(clay_color.expand_as(albedo))
            roughness.data.copy_(torch.full_like(roughness, 0.85))
            metallic.data.copy_(torch.full_like(metallic, 0.0))
            print("✅ 已切换为: 黏土 (Clay) - 粗糙表面光影测试")

        # ==========================================
        # 🧪 第三组：保留原有的相对属性修改
        # ==========================================
        elif edit_mode == "wet":
            # 淋雨/打湿：保留原颜色但变暗，粗糙度大幅降低
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
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
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
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
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
        edit_mode: str = "none",  # <--- 新增编辑模式参数
) -> None:
    iteration = scene.loaded_iter
    if name == "train":
        views = scene.getTrainCameras()
    elif name == "test":
        views = scene.getTestCameras()
    else:
        raise ValueError

    # build mip for environment light
    light.build_mips()
    envmap = light.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    os.makedirs(os.path.join(model_path, name), exist_ok=True)
    envmap_path = os.path.join(model_path, name, "envmap.png")
    torchvision.utils.save_image(envmap, envmap_path)

    # === 🔥 核心修改：动态输出路径 🔥 ===
    # 如果有编辑模式，放到 edit_xxx_30000 文件夹里，否则保持 ours_30000
    if edit_mode == "none":
        folder_name = f"ours_{iteration}"
    else:
        folder_name = f"edit_{edit_mode}_{iteration}"

    render_path = os.path.join(model_path, name, folder_name, "renders")
    gts_path = os.path.join(model_path, name, folder_name, "gt")
    depths_path = os.path.join(model_path, name, folder_name, "depth")
    normals_path = os.path.join(model_path, name, folder_name, "normal")
    pbr_path = os.path.join(model_path, name, folder_name, "pbr")
    pc_path = os.path.join(model_path, name, folder_name, "pc")

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
    view_dirs_ = (  # NOTE: no negative here
        (canonical_rays[:, None, :] * c2w[None, :3, :3]).sum(dim=-1).reshape(H, W, 3)  # [HW, 3, 3]
    )  # [H, W, 3]
    norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(H, W, 1)

    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0
    lpips_fn = LPIPS(net="vgg").cuda()

    for idx, view in enumerate(tqdm(views, desc=f"Rendering ({folder_name})")):
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        background2 = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
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
        c2w = torch.inverse(view.world_view_transform.T)  # [4, 4]
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]

        if indirect:
            occlusion = rendering_result["occlusion_map"].permute(1, 2, 0)
        else:
            occlusion = rendering_result["occlusion_map"].permute(1, 2, 0)

        torchvision.utils.save_image(
            (normal_map_from_depth + 1) / 2,
            os.path.join(normals_path, f"{idx:05d}_from_depth.png"),
        )

        if pbr:
            albedo_map = rendering_result["albedo_map"]  # [3, H, W]
            roughness_map = rendering_result["roughness_map"]  # [1, H, W]
            metallic_map = rendering_result["metallic_map"]  # [1, H, W]
            out_normal_view = rendering_result["out_normal_view"]
            depth_pos = rendering_result["depth_pos"]

            pbr_result = pbr_shading(
                light=light,
                normals=normal_map.permute(1, 2, 0),  # [H, W, 3]
                view_dirs=view_dirs,
                mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
                albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
                roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
                metallic=metallic_map.permute(1, 2, 0) if metallic else None,  # [H, W, 1]
                tone=tone,
                gamma=gamma,
                occlusion=occlusion,
                brdf_lut=brdf_lut,
            )
            render_rgb = (
                pbr_result["render_rgb"].permute(2, 0, 1)
            )  # [3, H, W]

            diffuse_rgb = (
                pbr_result["diffuse_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
            )  # [3, H, W]
            specular_rgb = (
                pbr_result["specular_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
            )

            render_rgb = torch.where(
                normal_mask,
                render_rgb,
                background[:, None, None]
            )
            diffuse_rgb = torch.where(
                normal_mask,
                diffuse_rgb,
                background[:, None, None]
            )

            specular_rgb = torch.where(
                normal_mask,
                specular_rgb,
                background[:, None, None]
            )

            SSR = Gaussian_SSR(tanfovx, tanfovy, image_width, image_height, radius, bias, thick, delta, step, start)
            if metallic:
                F0 = (1.0 - metallic) * 0.04 + albedo_map * metallic_map
            else:
                F0 = torch.ones_like(albedo_map) * 0.04  # [1, H, W, 3]
                metallic_map = torch.zeros_like(roughness_map)

            linear_rgb = srgb_to_linear(render_rgb)

            (IRR, _) = SSR(out_normal_view, depth_pos, linear_rgb, albedo_map, roughness_map, metallic_map, F0)
            IRR2 = IRR
            IRR2 = linear_to_srgb(IRR2)
            IRR = kornia.filters.median_blur(IRR[None, ...], (3, 3))[0]
            IRR2 = kornia.filters.median_blur(IRR2[None, ...], (3, 3))[0]

            render_rgb = render_rgb + IRR2
            render_rgb = torch.where(
                normal_mask,
                render_rgb,
                background[:, None, None]
            )

            albedo_map = (albedo_map * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
            roughness_map = (roughness_map * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0,
                                                                                                                1.0)
            metallic_map = (metallic_map * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

            bg_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
            normal_map = normal_map * alpha_mask + bg_normal[:, None, None] * (1.0 - alpha_mask)

            diffuse_rgb = (diffuse_rgb * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
            specular_rgb = (specular_rgb * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

            brdf_map = torch.cat(
                [
                    albedo_map,
                    roughness_map,
                    metallic_map
                ],
                dim=2,
            )
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

            # 🔥 保存核心渲染图到 renders 目录，方便查看
            torchvision.utils.save_image(render_rgb, os.path.join(render_path, f"{idx:05d}.png"))
            # PBR 目录也存一份（保留你的原始习惯）
            torchvision.utils.save_image(render_rgb, os.path.join(pbr_path, f"{idx:05d}.png"))

            torchvision.utils.save_image(diffuse_rgb, os.path.join(pbr_path, f"{idx:05d}_diffuse.png"))
            torchvision.utils.save_image(specular_rgb, os.path.join(pbr_path, f"{idx:05d}_specular.png"))
            torchvision.utils.save_image(render_rgb - IRR2, os.path.join(pbr_path, f"{idx:05d}_DIR.png"))
            torchvision.utils.save_image((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()),
                                         os.path.join(depths_path, f"{idx:05d}_depth.png"))
            torchvision.utils.save_image(IRR2, os.path.join(pbr_path, f"{idx:05d}_indirect.png"))

            # 评估模式下才计算 PSNR (如果是编辑模式，PSNR 会很低因为长得和 GT 完全不一样了，但不影响跑完)
            psnr_avg += get_psnr(gt_image, render_rgb).mean().double()
            ssim_avg += get_ssim(gt_image, render_rgb).mean().double()
            lpips_avg += lpips_fn(gt_image, render_rgb).mean().double()

    if pbr:
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
        edit_mode: str = "none",  # <--- 接收编辑模式
) -> None:
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    cubemap = CubemapLight(base_res=256).cuda()

    checkpoint = torch.load(checkpoint_path)
    model_params = checkpoint["gaussians"]
    cubemap_params = checkpoint["cubemap"]

    gaussians.restore(model_params)

    # === 🌟 材质编辑应用点 🌟 ===
    apply_material_editing(gaussians, edit_mode=edit_mode)

    cubemap.load_state_dict(cubemap_params)
    cubemap.eval()

    if brdf_eval:
        if not skip_train:
            eval_brdf(
                data_root=dataset.source_path,
                scene=scene,
                model_path=model_path,
                name="train",
                edit_mode=edit_mode
            )
        if not skip_test:
            eval_brdf(
                data_root=dataset.source_path,
                scene=scene,
                model_path=model_path,
                name="test",
                edit_mode=edit_mode
            )
    else:
        if not skip_train:
            render_set(
                model_path=model_path,
                name="train",
                scene=scene,
                light=cubemap,
                pipeline=pipeline,
                pbr=pbr,
                metallic=metallic,
                tone=tone,
                gamma=gamma,
                radius=radius,
                bias=bias,
                thick=thick,
                delta=delta,
                step=step,
                start=start,
                indirect=indirect,
                edit_mode=edit_mode  # <--- 传递给 render_set
            )
        if not skip_test:
            render_set(
                model_path=model_path,
                name="test",
                scene=scene,
                light=cubemap,
                pipeline=pipeline,
                pbr=pbr,
                metallic=metallic,
                tone=tone,
                gamma=gamma,
                radius=radius,
                bias=bias,
                thick=thick,
                delta=delta,
                step=step,
                start=start,
                indirect=indirect,
                edit_mode=edit_mode  # <--- 传递给 render_set
            )


def eval_brdf(data_root: str, scene: Scene, model_path: str, name: str, edit_mode: str = "none") -> None:
    # only for TensoIR synthetic
    if name == "train":
        transform_file = os.path.join(data_root, "transforms_train.json")
    elif name == "test":
        transform_file = os.path.join(data_root, "transforms_test.json")

    with open(transform_file, "r") as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]

    iteration = scene.loaded_iter

    # 兼容 Eval 路径
    if edit_mode == "none":
        folder_name = f"ours_{iteration}"
    else:
        folder_name = f"edit_{edit_mode}_{iteration}"

    pbr_dir = os.path.join(model_path, name, folder_name, "pbr")

    albedo_psnr_avg = 0.0
    albedo_ssim_avg = 0.0
    albedo_lpips_avg = 0.0
    mse_loss = 0.0

    pbr_path = os.path.join(model_path, name, folder_name, "pbr")
    albedo_gts = []
    albedo_maps = []
    masks = []
    gt_albedo_list = []
    reconstructed_albedo_list = []
    lpips_fn = LPIPS(net="vgg").cuda()
    mse = torch.nn.MSELoss(reduction='mean')
    for idx, frame in enumerate(tqdm(frames)):
        # read gt
        if "Synthetic4Relight" in data_root:
            albedo_path = frame["file_path"] + "_albedo.png"
        elif "orb" in data_root:
            albedo_path = frame["file_path"].replace("test", "pseudo_gt_albedo") + ".png"
            mask_path = frame["file_path"].replace("test", "test_mask") + ".png"
            data_root2 = data_root.replace("blender_LDR", "ground_truth")
        else:
            albedo_path = frame["file_path"].replace("rgba", "albedo") + ".png"

        if "orb" in data_root:
            albedo_gt = np.array(Image.open(os.path.join(data_root2, albedo_path)).resize((512, 512)))[..., :3]
        else:
            albedo_gt = np.array(Image.open(os.path.join(data_root, albedo_path)))[..., :3]

        if "orb" in data_root:
            mask = np.array(Image.open(os.path.join(data_root, mask_path)).resize((512, 512))) > 0
            expanded_mask = np.expand_dims(mask, axis=-1)
            mask_3d = np.repeat(expanded_mask, 3, axis=-1)
        else:
            mask = np.array(Image.open(os.path.join(data_root, albedo_path)))[..., 3] > 0
            expanded_mask = np.expand_dims(mask, axis=-1)
            mask_3d = np.repeat(expanded_mask, 3, axis=-1)

        albedo_gt[~mask_3d] = 0
        albedo_gt = torch.from_numpy(albedo_gt).cuda() / 255.0  # [H, W, 3]
        mask = torch.from_numpy(mask).cuda()  # [H, W]
        masks.append(mask)
        albedo_gts.append(albedo_gt)
        gt_albedo_list.append(albedo_gt[mask])

        # read prediction
        albedo_map = np.array(Image.open(os.path.join(pbr_dir, f"{idx:05}_albedo.png")))[..., :3]
        albedo_map[~mask_3d] = 0
        albedo_map = torch.from_numpy(albedo_map).cuda() / 255.0  # [H, W, 3]
        albedo_maps.append(albedo_map)
        reconstructed_albedo_list.append(albedo_map[mask])

    gt_albedo_all = torch.cat(gt_albedo_list, dim=0)
    albedo_map_all = torch.cat(reconstructed_albedo_list, dim=0)

    for idx, (mask, albedo_map, albedo_gt) in enumerate(tqdm(zip(masks, albedo_maps, albedo_gts))):
        roughmse = (albedo_map - albedo_gt) ** 2  # 平方误差
        masked_diff = roughmse[mask]
        mse_loss += masked_diff.mean()

    roughmse = mse_loss / len(frames)
    print(f"roughmse: {roughmse}")


if __name__ == "__main__":
    # Set up command line argument parser
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

    # === 🔥 增加材质编辑命令参数 🔥 ===
    parser.add_argument("--edit_mode", type=str, default="none",
                        choices=["none", "gold", "copper", "silver", "plastic", "rubber", "obsidian", "clay", "wet",
                                 "ceramic"],
                        help="Material editing mode. Output will be saved in a separate folder.")

    args = get_combined_args(parser)

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)

    # Initialize system state (RNG)
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
    )