import os
import argparse
import cv2
import numpy as np
from pathlib import Path
#uint8转为uint16

def convert_uint8_to_uint16(input_dir, output_dir=None, scale_mode='stretch'):
    """
    将文件夹内的 uint8 深度图转换为 uint16。

    参数:
    input_dir (str): 输入文件夹路径
    output_dir (str): 输出文件夹路径。如果为 None，默认在同级目录下创建 'output_uint16'
    scale_mode (str):
        - 'stretch': 线性拉伸 (0->0, 255->65535). 适用于可视化扩展。
        - 'cast': 直接类型转换 (0->0, 255->255). 适用于数值本身很小的情况。
        - 'shift': 移位操作 (value << 8). 类似于 value * 256.
    """

    # 检查输入路径
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入文件夹 '{input_dir}' 不存在。")
        return

    # 设置输出路径
    if output_dir is None:
        output_path = input_path.parent / (input_path.name + "_uint16")
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path}")

    # 支持的图片扩展名
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    count = 0

    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            # 1. 读取图片 (以灰度模式读取)
            img_uint8 = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

            if img_uint8 is None:
                print(f"警告: 无法读取文件 {file_path.name}")
                continue

            # 确保是单通道
            if len(img_uint8.shape) > 2:
                img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

            # 检查原始数据类型
            if img_uint8.dtype != np.uint8:
                print(f"跳过: {file_path.name} 不是 uint8 格式 (当前是 {img_uint8.dtype})")
                continue

            # 2. 转换逻辑
            # uint8 范围: 0 - 255
            # uint16 范围: 0 - 65535

            if scale_mode == 'stretch':
                # 线性拉伸: pixel * (65535 / 255) ≈ pixel * 257
                img_uint16 = (img_uint8.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
            elif scale_mode == 'shift':
                # 左移 8 位: pixel * 256
                img_uint16 = (img_uint8.astype(np.uint16) << 8)
            else:  # 'cast'
                # 纯粹的类型转换，数值不变
                img_uint16 = img_uint8.astype(np.uint16)

            # 3. 保存图片
            # 注意：保存为 PNG 格式，因为 JPG 不支持 16 位
            save_name = file_path.stem + ".png"
            save_path = output_path / save_name

            cv2.imwrite(str(save_path), img_uint16)
            count += 1
            print(f"[{count}] 已转换: {file_path.name} -> {save_name}")

    print(f"\n处理完成! 共转换 {count} 张图片。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将文件夹内的 uint8 深度图批量转换为 uint16")

    parser.add_argument("--input_dir","-i", type=str, help="存放 uint8 图片的输入文件夹路径")
    parser.add_argument("--output_dir","-o", type=str, default=None, help="输出文件夹路径 (默认: 输入文件夹旁边的 *_uint16)")
    parser.add_argument("--mode", type=str, choices=['stretch', 'shift', 'cast'], default='stretch',
                        help="转换模式: 'stretch' (拉伸0-65535, 默认), 'shift' (左移8位), 'cast' (仅改变类型数值不变)")

    args = parser.parse_args()

    convert_uint8_to_uint16(args.input_dir, args.output_dir, args.mode)