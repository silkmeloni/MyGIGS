import os
import shutil
import argparse
from tqdm import tqdm  # 如果没有安装 tqdm，可以使用 pip install tqdm 安装，或者删除相关代码
"""
    处理Blender数据集，将其分割成单独的文件夹
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Reorganize Blender dataset into separate folders.")

    # 原数据集路径
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the original dataset (containing train/ and test/ folders)")

    # 输出路径 (默认为当前目录下的 dataset_processed)
    parser.add_argument("--output_path", type=str, default="./dataset_processed",
                        help="Path where the organized data will be saved")

    return parser.parse_args()


def organize_split(source_root, output_root, split_name):
    """
    处理单个 split (train 或 test)
    """
    source_dir = os.path.join(source_root, split_name)

    # 如果源文件夹不存在，直接跳过
    if not os.path.exists(source_dir):
        print(f"Warning: Source folder {source_dir} does not exist. Skipping.")
        return

    # 定义目标文件夹名称
    # 目标结构: output/train_single/rgb, output/train_single/depth, ...
    target_base = os.path.join(output_root, f"{split_name}_single")
    target_rgb = os.path.join(target_base, "rgb")
    target_depth = os.path.join(target_base, "depth")
    target_normal = os.path.join(target_base, "normal")

    # 创建目标文件夹
    os.makedirs(target_rgb, exist_ok=True)
    os.makedirs(target_depth, exist_ok=True)
    os.makedirs(target_normal, exist_ok=True)

    print(f"Processing {split_name} data...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_base}")

    files = os.listdir(source_dir)

    # 使用 tqdm 显示进度条
    for filename in tqdm(files, desc=f"Organizing {split_name}"):
        src_file = os.path.join(source_dir, filename)

        # 忽略文件夹，只处理文件
        if not os.path.isfile(src_file):
            continue

        # --- 分类逻辑 ---

        # 1. 深度图 (_disp.tiff)
        if filename.endswith("_disp.tiff") or filename.endswith("_disp.png"):
            dst_file = os.path.join(target_depth, filename)
            shutil.copy2(src_file, dst_file)

        # 2. 法线图 (_normal.png)
        elif filename.endswith("_normal.png"):
            dst_file = os.path.join(target_normal, filename)
            shutil.copy2(src_file, dst_file)

        # 3. RGB图 (排除掉法线图后的 .png)
        # 注意：必须放在 _normal.png 判断之后，因为 _normal.png 也以 .png 结尾
        elif filename.endswith(".png"):
            dst_file = os.path.join(target_rgb, filename)
            shutil.copy2(src_file, dst_file)

    print(f"Done processing {split_name}.\n")


def main():
    args = parse_args()

    if not os.path.exists(args.source_path):
        print(f"Error: Source path '{args.source_path}' does not exist.")
        return

    # 处理 train 文件夹
    organize_split(args.source_path, args.output_path, "train")

    # 处理 test 文件夹
    organize_split(args.source_path, args.output_path, "test")

    print(f"All done! Organized dataset saved to: {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()