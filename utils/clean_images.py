import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="修复图片格式：强制转换为 8-bit RGB，移除 Alpha 通道，修复 COLMAP 读取错误")

    parser.add_argument("--source_path", "-s", type=str, required=True,
                        help="数据集的根目录路径 (脚本会递归搜索该目录下的所有 png 图片)")

    return parser.parse_args()


def sanitize_image(img_path):
    try:
        # 1. 读取图片 (IMREAD_UNCHANGED 保留原始格式以便检查)
        # OpenCV 的读取非常鲁棒，能处理各种奇怪的文件头
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[Error] 无法读取文件，可能已损坏: {img_path}")
            return False

        original_shape = img.shape
        original_dtype = img.dtype
        need_save = False

        # # --- 检查 1: 通道数 (移除 Alpha 通道) ---
        # # 如果是 (H, W, 4)，说明是 RGBA
        # if len(img.shape) == 3 and img.shape[2] == 4:
        #     # 只保留前三个通道 (BGR)
        #     img = img[:, :, :3]
        #     need_save = True

        # --- 检查 2: 位深 (16-bit 转 8-bit) ---
        if img.dtype == np.uint16:
            # 16位转8位 (除以256)
            img = (img / 256.0).astype(np.uint8)
            need_save = True
        elif img.dtype != np.uint8:
            # 其他非 uint8 类型 (如 float)，强制归一化转 uint8
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            need_save = True

        # --- 强制重写 ---
        # 即使上述检查都通过，为了修复潜在的文件头(Header)错误
        # 我们对所有图片都进行一次重写，确保 COLMAP 能够识别
        # (为了效率，如果你确定只有部分有问题，可以把下面的 True 改为 need_save)
        force_rewrite = True

        if force_rewrite or need_save:
            # 覆盖原文件
            cv2.imwrite(img_path, img)
            return True

    except Exception as e:
        print(f"[Exception] 处理 {img_path} 时出错: {e}")
        return False

    return False


def main():
    args = parse_args()

    root_dir = args.source_path

    if not os.path.exists(root_dir):
        print(f"Error: 路径不存在 -> {root_dir}")
        return

    print(f"正在扫描目录: {root_dir} ...")

    # 递归查找所有 .png 文件
    # recursive=True 配合 ** 可以通过 glob 遍历子文件夹
    pattern = os.path.join(root_dir, "**", "*.png")
    image_files = glob.glob(pattern, recursive=True)

    if len(image_files) == 0:
        print("未找到任何 .png 图片。")
        return

    print(f"找到 {len(image_files)} 张图片，开始清洗...")

    count_fixed = 0
    for img_path in tqdm(image_files):
        if sanitize_image(img_path):
            count_fixed += 1

    print("-" * 30)
    print(f"处理完成！")
    print(f"共扫描: {len(image_files)}")
    print(f"已修复/重写: {count_fixed}")
    print("现在 COLMAP 应该可以正常读取了。")


if __name__ == "__main__":
    main()