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

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import cv2  # [新增] 用于读取深度图计算 scale
import struct
from joblib import Parallel, delayed

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int
    depth_mono_path: str #ljx，da2的单目深度路径
    # [新增]
    depth_scale: float = 1.0
    depth_shift: float = 0.0


class SceneInfo(NamedTuple):
    point_cloud: Optional[BasicPointCloud]
    train_cameras: List
    test_cameras: List
    nerf_normalization: Dict
    ply_path: str


def getNerfppNorm(cam_info: List[CameraInfo]) -> Dict:
    def get_center_and_diag(cam_centers: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


# [新增] 读取 points3D 到字典的辅助函数 (Binary)
def read_points3D_binary_dict(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num_points):
            binary_point_line_properties = fid.read(43)
            point3D_id = struct.unpack("Q", binary_point_line_properties[0:8])[0]
            xyz = struct.unpack("3d", binary_point_line_properties[8:32])
            rgb = struct.unpack("3B", binary_point_line_properties[32:35])
            error = struct.unpack("d", binary_point_line_properties[35:43])[0]
            track_length = struct.unpack("Q", fid.read(8))[0]
            track_elems = fid.read(8 * track_length)
            points3D[point3D_id] = np.array(xyz)
    return points3D


# [新增] 读取 points3D 到字典的辅助函数 (Text)
def read_points3D_text_dict(path):
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                points3D[point3D_id] = xyz
                # Skip the next line (track information)
                fid.readline()
    return points3D


# [新增] 单张图片计算 Scale 和 Offset 的逻辑
def compute_depth_scale(key, cam_extrinsics, cam_intrinsics, points3d_dict, depths_dir):
    image_meta = cam_extrinsics[key]
    cam_intrinsic = cam_intrinsics[image_meta.camera_id]

    # 获取 2D 点和对应的 3D 点 ID
    pts_idx = image_meta.point3D_ids
    valid_xys = image_meta.xys

    # 筛选有效的 3D 点 (ID != -1 且在 points3d_dict 中存在)
    valid_mask = (pts_idx != -1)
    # 进一步检查 ID 是否真正在 dict 中 (防止 colmap 数据不一致)
    final_mask = []
    for i, pid in enumerate(pts_idx):
        if valid_mask[i] and pid in points3d_dict:
            final_mask.append(True)
        else:
            final_mask.append(False)
    final_mask = np.array(final_mask)

    if final_mask.sum() == 0:
        return None

    pts_idx = pts_idx[final_mask]
    valid_xys = valid_xys[final_mask]

    # 获取 3D 坐标
    pts = np.array([points3d_dict[pid] for pid in pts_idx])

    # 变换到相机坐标系
    R = qvec2rotmat(image_meta.qvec)
    pts_cam = np.dot(pts, R.T) + image_meta.tvec

    # 计算 COLMAP 深度 (z) 和 逆深度
    # 注意: 这里计算的是 z_depth, 你的脚本用的是 1. / z
    colmap_depth = pts_cam[..., 2]
    invcolmapdepth = 1.0 / (colmap_depth + 1e-7)

    # 读取单目深度图
    image_basename = os.path.basename(image_meta.name)
    image_name_no_ext = os.path.splitext(image_basename)[0]
    depth_path = os.path.join(depths_dir, f"{image_name_no_ext}.png")

    invmonodepthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if invmonodepthmap is None:
        return None

    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    # 归一化 Disparity (根据你的脚本 / 2**16)
    invmonodepthmap = invmonodepthmap.astype(np.float32) / 65535.0

    # 缩放因子 (适应不同分辨率)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    # 将 COLMAP 投影点映射到 深度图 像素坐标
    maps = (valid_xys * s).astype(np.float32)

    # 边界检查
    h_mono, w_mono = invmonodepthmap.shape
    valid = (
            (maps[..., 0] >= 0) * (maps[..., 1] >= 0) * (maps[..., 0] < w_mono) * (maps[..., 1] < h_mono) * (
                invcolmapdepth > 0)
    )

    scale = 0
    offset = 0

    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]

        # 采样单目深度值
        # cv2.remap 需要 map_x, map_y，这里是稀疏点，直接用 map_coordinates 或者 bilinear sampling
        # 为了方便，这里用 cv2.remap 的方式 (类似你原脚本) 或者直接取整/双线性插值
        # 你的脚本使用了 remap 这种 trick，这里我们为了简单且不引入 map_x 全图，可以使用 bilinear 采样函数
        # 但为了保持和你脚本完全一致，我们构建 map
        # 注意: cv2.remap 通常处理整张图。对于稀疏点，直接用 opencv 获取值比较麻烦
        # 这里为了效率，直接用 nearest 或者简单的 bilinear

        # 简单 bilinear 插值实现
        x = maps[:, 0]
        y = maps[:, 1]
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, w_mono - 1)
        x1 = np.clip(x1, 0, w_mono - 1)
        y0 = np.clip(y0, 0, h_mono - 1)
        y1 = np.clip(y1, 0, h_mono - 1)

        Ia = invmonodepthmap[y0, x0]
        Ib = invmonodepthmap[y1, x0]
        Ic = invmonodepthmap[y0, x1]
        Id = invmonodepthmap[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        invmonodepth = wa * Ia + wb * Ib + wc * Ic + wd * Id

        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))

        if s_mono > 1e-7:
            scale = s_colmap / s_mono
            offset = t_colmap - t_mono * scale

    return {"image_name": image_name_no_ext, "scale": scale, "offset": offset}

def readColmapCameras(
    cam_extrinsics: Dict, cam_intrinsics: Dict, images_folder: str,use_mono_depth=False,depth_params=None
) -> List[CameraInfo]:
    print("执行了readColmapCameras")
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write(f"Reading camera {idx + 1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        #ljx:读取da2单目深度图
        depth_mono_path = None  # 默认为 None
        if use_mono_depth:  # 只有开启了参数，才去拼凑路径
            depth_folder = os.path.join(os.path.dirname(images_folder), "da2")
            image_basename = os.path.basename(extr.name)
            image_name_no_ext = os.path.splitext(image_basename)[0]
            depth_mono_path = os.path.join(depth_folder, f"{image_name_no_ext}.png")
            # =================================

            if not os.path.exists(depth_mono_path):
                # 还可以加一层容错：如果 png 找不到，尝试找同名 jpg (虽然一般深度图都是png)
                depth_mono_path_jpg = os.path.join(depth_folder, f"{image_name_no_ext}.jpg")
                if os.path.exists(depth_mono_path_jpg):
                    depth_mono_path = depth_mono_path_jpg
                else:
                    print(f"Warning: Depth map not found at {depth_mono_path}")
                    depth_mono_path = None  # 没找到务必设为 None
            else:
                # print(f"find Depth map at {depth_mono_path}!")
                pass

        #读取深度图尺度偏移
        # [新增] 从 depth_params 字典中获取 scale 和 offset
        # 默认值设为 1.0 和 0.0 (即不对齐)
        d_scale, d_shift = 1.0, 0.0
        if depth_params is not None:
        # 你的脚本生成的 json key 是不带后缀的文件名，正好和 image_name 对应
            if image_name in depth_params:
                d_scale = depth_params[image_name]["scale"]
                d_shift = depth_params[image_name]["offset"]
            elif use_mono_depth:
                print(f"[Warning] No depth params found for {image_name}")



        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            depth_mono_path=depth_mono_path,  # <--- 传入路径
            depth_scale=d_scale,
            depth_shift=d_shift
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path: str) -> BasicPointCloud:
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path: str, images: str, eval: bool, llffhold: int = 8,use_mono_depth=False) -> SceneInfo:
    print("执行了readColmapSceneInfo")
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    #读取 sparse/0/depth_params.json
    # [新增] 自动检测和计算 Depth Scale/Offset
    depth_params = None
    reading_dir = "images" if images == None else images
    images_folder = os.path.join(path, reading_dir)

    if use_mono_depth:
        json_path = os.path.join(path, "sparse/0/depth_params.json")

        # 1. 尝试读取
        if os.path.exists(json_path):
            print(f"Loading pre-computed depth params from {json_path}")
            with open(json_path, 'r') as f:
                depth_params = json.load(f)
        else:
            # 2. 如果不存在，自动计算
            print(f"[Info] {json_path} not found. Automatically calculating depth scales...")
            depths_dir = os.path.join(os.path.dirname(images_folder), "da2")
            if not os.path.exists(depths_dir):
                print(f"[Error] Depth directory {depths_dir} not found! Cannot align depths.")
            else:
                # 需要读取 3D 点云字典来做对齐
                points3d_bin = os.path.join(path, "sparse/0", "points3D.bin")
                points3d_txt = os.path.join(path, "sparse/0", "points3D.txt")

                points3d_dict = None
                if os.path.exists(points3d_bin):
                    points3d_dict = read_points3D_binary_dict(points3d_bin)
                elif os.path.exists(points3d_txt):
                    points3d_dict = read_points3D_text_dict(points3d_txt)

                if points3d_dict is not None:
                    # 并行计算 (使用 joblib)
                    print("Computing alignment in parallel...")
                    results = Parallel(n_jobs=-1, backend="threading")(
                        delayed(compute_depth_scale)(
                            key, cam_extrinsics, cam_intrinsics, points3d_dict, depths_dir
                        ) for key in cam_extrinsics
                    )

                    # 整理结果
                    depth_params = {}
                    for res in results:
                        if res is not None:
                            depth_params[res["image_name"]] = {"scale": res["scale"], "offset": res["offset"]}

                    # 保存 JSON
                    print(f"Saving computed params to {json_path}")
                    with open(json_path, "w") as f:
                        json.dump(depth_params, f, indent=2)
                else:
                    print("[Error] Could not read points3D.bin/.txt to compute scales.")




    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        use_mono_depth=use_mono_depth,
        depth_params=depth_params
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(
    path: str, transformsfile: str, white_background: bool, extension: str = ".png"
) -> List[CameraInfo]:
    print("执行了readCamerasFromTransforms")
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        cam_name = os.path.join(path, frame["file_path"] + extension)

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        # im_data = np.array(image.convert("RGBA"))

        # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        # norm_data = im_data / 255.0
        # arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        # image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy
        FovX = fovx

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
                depth_mono_path=None,  # <--- 传入路径
            )
        )

    return cam_infos


def readNerfSyntheticInfo(
    path: str, white_background: bool, eval: bool, extension: str = ".png"
) -> SceneInfo:
    print("执行了readNerfSyntheticInfo")
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {"Colmap": readColmapSceneInfo, "Blender": readNerfSyntheticInfo}