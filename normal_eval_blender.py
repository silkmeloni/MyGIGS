import os
import glob
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_mae(gt_normal_stack: np.ndarray, render_normal_stack: np.ndarray) -> float:
    # compute mean angular error
    # clip at -1, 1 to avoid nan in arccos
    MAE = np.mean(
        np.arccos(np.clip(np.sum(gt_normal_stack * render_normal_stack, axis=-1), -1, 1))
        * 180
        / np.pi
    )
    return MAE.item()


if __name__ == "__main__":
    parser = ArgumentParser(description="Normal evaluation script")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory (e.g., .../output/point_cloud/iteration_30000)")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Path to the dataset root (containing 'test' folder)")
    args = parser.parse_args()

    # 1. 确定 GT 文件夹路径 (args.gt_dir/test)
    gt_test_dir = os.path.join(args.gt_dir, "test")
    if not os.path.exists(gt_test_dir):
        # 容错：有些数据集直接传的就是 test 文件夹路径
        if os.path.basename(args.gt_dir) == "test":
            gt_test_dir = args.gt_dir
        else:
            raise FileNotFoundError(f"GT test folder not found at {gt_test_dir}")

    # 2. 获取所有 GT 法线文件 (r_*_normal.png)
    # 使用 glob 匹配 r_{id}_normal.png
    gt_files = glob.glob(os.path.join(gt_test_dir, "r_*_normal.png"))
    gt_files.sort()  # 排序确保顺序一致

    if len(gt_files) == 0:
        print(f"No normal files found in {gt_test_dir}. Pattern: r_*_normal.png")
        exit(1)

    print(f"Found {len(gt_files)} GT normal maps in {gt_test_dir}")

    normal_gt_stack = []
    normal_gs_stack = []
    normal_from_depth_stack = []

    # 默认背景法线 (指向 Z 轴)
    normal_bg = np.array([0.0, 0.0, 1.0])

    for gt_path in tqdm(gt_files, desc="Evaluating"):
        filename = os.path.basename(gt_path)

        # 解析 ID: "r_0_normal.png" -> 0
        try:
            # 移除后缀 "_normal.png" 得到 "r_0"
            # 再移除前缀 "r_" 得到 "0"
            base_name = filename.replace("_normal.png", "")
            test_id = int(base_name.split("_")[-1])
        except ValueError:
            print(f"Skipping {filename}, cannot parse ID.")
            continue

        # --- 读取 GT ---
        normal_gt_img = Image.open(gt_path)
        normal_gt = np.array(normal_gt_img)

        # 归一化并从 [0, 1] 映射到 [-1, 1]
        normal_gt_rgb = normal_gt[..., :3] / 255.0
        normal_gt_vec = (normal_gt_rgb - 0.5) * 2.0

        # 处理 Alpha Mask (如果有)
        if normal_gt.shape[-1] == 4:
            alpha_mask = normal_gt[..., [-1]] / 255.0
            normal_gt_vec = normal_gt_vec * alpha_mask + normal_bg * (1.0 - alpha_mask)

        # 归一化向量长度
        normal_gt_vec = normal_gt_vec / np.linalg.norm(normal_gt_vec, axis=-1, ord=2, keepdims=True)
        normal_gt_stack.append(normal_gt_vec)

        # --- 读取 Prediction (GS Normal) ---
        # Output 格式假设为: output_dir/normal/{05d}_normal.png
        pred_filename = f"{test_id:05d}_normal.png"
        normal_gs_path = os.path.join(args.output_dir, "normal", pred_filename)

        if not os.path.exists(normal_gs_path):
            print(f"Warning: Prediction not found: {normal_gs_path}")
            # 这里可以选择 continue 或者补一个 dummy 数据，为了对齐最好 crash 或者跳过 GT
            # 为了简单起见，这里我们把刚刚加进去的 GT 吐出来，并跳过这次循环
            normal_gt_stack.pop()
            continue

        normal_gs_img = Image.open(normal_gs_path)
        normal_gs = np.array(normal_gs_img)[..., :3] / 255.0
        normal_gs = (normal_gs - 0.5) * 2.0

        # Trick: 处理背景色 (128/255, 128/255, 1.0) 也就是淡蓝色
        # 你的 output 代码里可能把背景设为了这个颜色
        # 检查是否为淡蓝色背景 (128, 128, 255)
        is_bg = (np.array(normal_gs_img)[..., :3] == np.array([128, 128, 255], dtype=np.uint8)).all(-1)
        normal_gs[is_bg] = normal_bg

        normal_gs = normal_gs / np.linalg.norm(normal_gs, axis=-1, ord=2, keepdims=True)
        normal_gs_stack.append(normal_gs)

        # --- 读取 Prediction (Normal from Depth) ---
        # 可选：如果不需要评测这个，可以注释掉
        pred_depth_filename = f"{test_id:05d}_from_depth.png"
        normal_depth_path = os.path.join(args.output_dir, "normal", pred_depth_filename)

        if os.path.exists(normal_depth_path):
            normal_depth_img = Image.open(normal_depth_path)
            normal_depth = np.array(normal_depth_img)[..., :3] / 255.0
            normal_depth = (normal_depth - 0.5) * 2.0

            is_bg_depth = (np.array(normal_depth_img)[..., :3] == np.array([128, 128, 255], dtype=np.uint8)).all(-1)
            normal_depth[is_bg_depth] = normal_bg

            normal_depth = normal_depth / np.linalg.norm(normal_depth, axis=-1, ord=2, keepdims=True)
            normal_from_depth_stack.append(normal_depth)

    # 计算 MAE
    if len(normal_gt_stack) > 0:
        normal_gt_stack = np.stack(normal_gt_stack)
        normal_gs_stack = np.stack(normal_gs_stack)

        print(f"Evaluating {len(normal_gt_stack)} images...")
        mae_gs = get_mae(normal_gt_stack, normal_gs_stack)
        print(f"MAE (Predicted Normal): {mae_gs:.4f}")

        if len(normal_from_depth_stack) == len(normal_gt_stack):
            normal_from_depth_stack = np.stack(normal_from_depth_stack)
            mae_from_depth = get_mae(normal_gt_stack, normal_from_depth_stack)
            print(f"MAE (Normal from Depth): {mae_from_depth:.4f}")
    else:
        print("No valid image pairs found.")