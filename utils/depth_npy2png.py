import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="将 float32 格式的 .npy 深度图转换为 uint16 格式的 .png")

    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="包含 .npy 文件的输入文件夹路径")

    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="保存 .png 文件的输出文件夹路径")

    parser.add_argument("--scale", "-s", type=float, default=1000.0,
                        help="缩放因子：将米转换为整数。默认为 1000.0 (即 1000 = 1米，单位为毫米)。保存为 uint16。")

    return parser.parse_args()


def main():
    args = parse_args()

    # 检查输入文件夹是否存在
    if not os.path.exists(args.input_dir):
        print(f"Error: 输入文件夹不存在 -> {args.input_dir}")
        return

    # 创建输出文件夹
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有 npy 文件
    pattern = os.path.join(args.input_dir, "*.npy")
    npy_files = glob.glob(pattern)

    if len(npy_files) == 0:
        print(f"Warning: 在 {args.input_dir} 中没有找到 .npy 文件")
        return

    print(f"正在处理 {len(npy_files)} 个文件...")
    print(f"输入: {args.input_dir}")
    print(f"输出: {args.output_dir}")
    print(f"缩放因子: {args.scale} (请确保你在读取 PNG 时除以这个数值)")

    for npy_path in tqdm(npy_files):
        try:
            # 1. 读取 npy (假设单位是米, float32)
            depth_map = np.load(npy_path)

            # 2. 处理无效值 (inf, nan) -> 0
            depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0)

            # 3. 缩放并转换为 uint16
            # 例如: 2.5m * 1000 = 2500
            depth_scaled = depth_map * args.scale

            # 截断超过 uint16 范围的值 (65535)
            depth_scaled = np.clip(depth_scaled, 0, 65535)
            depth_uint16 = depth_scaled.astype(np.uint16)

            # 4. 生成保存路径
            # r_0.npy -> r_0.png
            file_name = os.path.basename(npy_path).replace(".npy", ".png")
            save_path = os.path.join(args.output_dir, file_name)

            # 5. 保存
            cv2.imwrite(save_path, depth_uint16)

        except Exception as e:
            print(f"转换失败 {npy_path}: {e}")

    print("转换全部完成！")


if __name__ == "__main__":
    main()