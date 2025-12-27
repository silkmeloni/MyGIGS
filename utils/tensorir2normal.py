import os
import shutil
import json
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="将分文件夹存储的 Blender 数据集扁平化为 3DGS 标准格式 (提取 RGB 和 Normal)")

    parser.add_argument("--source_path", "-s", type=str, required=True,
                        help="数据集的根目录路径 (包含 transforms_train.json 的文件夹)")

    return parser.parse_args()


def flatten_split(root_path, split):
    """
    处理单个 split (train 或 test)
    """
    json_filename = f"transforms_{split}.json"
    json_path = os.path.join(root_path, json_filename)

    if not os.path.exists(json_path):
        print(f"[Skip] {split} set not found: {json_filename} does not exist.")
        return

    print(f"=== Processing {split} set ===")

    # 1. 创建输出文件夹
    # RGB 输出目录: e.g., dataset/train
    out_rgb_dir = os.path.join(root_path, split)
    # Normal 输出目录: e.g., dataset/train_normal
    out_normal_dir = os.path.join(root_path, f"{split}_normal")

    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_normal_dir, exist_ok=True)

    # 2. 读取原始 JSON
    with open(json_path, 'r') as f:
        meta = json.load(f)

    new_frames = []
    normal_count = 0

    print(f"Source: {root_path}")
    print(f"Target RGB: {out_rgb_dir}")
    print(f"Target Normal: {out_normal_dir}")

    for i, frame in enumerate(tqdm(meta['frames'])):
        # 原始路径通常是: "./train_000/rgba" 或 "train_000/rgba" (没有后缀)
        old_file_path_raw = frame['file_path']

        # 修正路径：移除开头的 ./ (如果有)
        clean_path = old_file_path_raw.replace("./", "")

        # --- 寻找源 RGB 图片 ---
        # 假设结构是: root/train_000/rgba.png
        # 先尝试 .png
        src_rgb_path = os.path.join(root_path, clean_path + ".png")
        if not os.path.exists(src_rgb_path):
            # 尝试直接拼接 (万一 json 里自带后缀)
            src_rgb_path = os.path.join(root_path, clean_path)

        if not os.path.exists(src_rgb_path):
            # 如果还是找不到，尝试 jpg
            src_rgb_path_jpg = os.path.join(root_path, clean_path + ".jpg")
            if os.path.exists(src_rgb_path_jpg):
                src_rgb_path = src_rgb_path_jpg
            else:
                print(f"[Warn] Image not found for frame {i}: {clean_path}")
                continue

        # --- 寻找源 Normal 图片 ---
        # 获取所在的父文件夹 (例如 train_000)
        parent_dir = os.path.dirname(src_rgb_path)

        # 假设法线就在 train_000/normal.png
        src_normal_path = os.path.join(parent_dir, "normal.png")
        # 有些数据集可能叫 normals.png 或 normal_map.png，可以在这里加判断

        # --- 定义新文件名 (统一重命名) ---
        # 格式: r_0.png, r_1.png ...
        extension = os.path.splitext(src_rgb_path)[1]  # 获取原后缀 (.png)
        new_filename = f"r_{i}{extension}"

        # 3. 复制 RGB
        dst_rgb_path = os.path.join(out_rgb_dir, new_filename)
        shutil.copy(src_rgb_path, dst_rgb_path)

        # 4. 复制 Normal (如果存在)
        if os.path.exists(src_normal_path):
            # 法线通常保存为 png 以保证精度
            dst_normal_path = os.path.join(out_normal_dir, f"r_{i}.png")
            shutil.copy(src_normal_path, dst_normal_path)
            normal_count += 1

        # 5. 更新 Frame 信息
        # 新路径: ./train/r_0 (注意不带后缀，这是 NeRF/3DGS 格式习惯)
        frame['file_path'] = f"./{split}/r_{i}"
        new_frames.append(frame)

    # 6. 保存新 JSON
    # 命名为 transforms_train_standard.json 以免覆盖原始文件
    meta['frames'] = new_frames
    new_json_path = os.path.join(root_path, f"transforms_{split}_standard.json")
    with open(new_json_path, 'w') as f:
        json.dump(meta, f, indent=4)

    print(f"Split '{split}' done.")
    print(f"  - RGB Saved: {len(new_frames)}")
    print(f"  - Normals Saved: {normal_count}")
    print(f"  - New JSON: {new_json_path}\n")


def main():
    args = parse_args()

    if not os.path.exists(args.source_path):
        print(f"Error: Path does not exist -> {args.source_path}")
        return

    # 分别处理 train 和 test
    flatten_split(args.source_path, "train")
    flatten_split(args.source_path, "test")

    # 如果有 val 也顺便处理
    flatten_split(args.source_path, "val")


if __name__ == "__main__":
    main()