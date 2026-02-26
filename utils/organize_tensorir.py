import os
import shutil
import argparse
from tqdm import tqdm


def organize_rgba_strict(input_root, output_root):
    # 1. 定义输出路径
    output_train = os.path.join(output_root, "train")
    output_test = os.path.join(output_root, "test")

    # 2. 创建输出目录
    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_test, exist_ok=True)

    print(f"输入路径: {input_root}")
    print(f"输出路径: {output_root}")
    print("-" * 40)

    if not os.path.exists(input_root):
        print("错误：输入路径不存在！")
        return

    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]

    success_count = 0

    # 3. 遍历子文件夹
    for folder_name in tqdm(subfolders, desc="Processing"):
        src_folder_path = os.path.join(input_root, folder_name)

        # --- 判定归属 (Train / Test) ---
        target_dir = None
        if folder_name.startswith("train"):
            target_dir = output_train
        elif folder_name.startswith("test"):
            target_dir = output_test

        # 如果既不是 train 开头也不是 test 开头，直接忽略
        if target_dir is None:
            continue

        # --- 锁定目标文件: rgba.png ---
        src_file_path = os.path.join(src_folder_path, "rgba.png")

        # 只有当 rgba.png 存在时才处理
        if os.path.exists(src_file_path):
            # --- 严格命名逻辑 ---
            # 格式: [文件夹名]_rgba.png
            # 例如: train_000 -> train_000_rgba.png
            # 这样保证了没有其他多余后缀
            new_filename = f"{folder_name}_rgba.png"

            dst_file_path = os.path.join(target_dir, new_filename)

            try:
                shutil.copy2(src_file_path, dst_file_path)
                success_count += 1
            except Exception as e:
                print(f"[复制出错] {src_file_path} -> {e}")

    print("-" * 40)
    print(f"处理完成！")
    print(f"共提取并重命名: {success_count} 张图片")
    print(f"Train 输出目录: {output_train}")
    print(f"Test  输出目录: {output_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract only 'rgba.png' and rename strictly to 'foldername_rgba.png'.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to source folder (e.g., contains train_000, test_001)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to destination folder")

    args = parser.parse_args()

    organize_rgba_strict(args.input_path, args.output_path)