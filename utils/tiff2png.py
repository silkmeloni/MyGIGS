import os
import cv2
import argparse
import numpy as np
import sys
from tqdm import tqdm  # 进度条库，如果没有安装可 pip install tqdm


def convert_tiff_to_png(input_dir, output_dir):
    # 1. 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误: 输入文件夹 '{input_dir}' 不存在。")
        sys.exit(1)

    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print(f"输入路径: {input_dir}")
    print(f"输出路径: {output_dir}")
    print("-" * 40)

    # 支持的 TIFF 后缀
    valid_exts = {'.tiff', '.tif'}

    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    if not files:
        print("警告: 输入文件夹中没有找到 .tiff 或 .tif 文件。")
        return

    success_count = 0
    fail_count = 0

    for filename in tqdm(files, desc="转换中"):
        src_path = os.path.join(input_dir, filename)

        # 3. 读取 TIFF (关键: flags=-1 确保读取原始深度/视差数据)
        # 这样可以保留 16-bit (uint16) 或 32-bit (float) 的精度
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"\n[读取失败] 无法读取: {filename}")
            fail_count += 1
            continue

        # 4. 数据类型处理 (防止 PNG 不支持 float32)
        # PNG 标准通常支持 8-bit 和 16-bit (uint16)。
        # 如果源 TIFF 是 32-bit float，我们需要把它转成 16-bit uint16 才能保存为标准 PNG。
        # 这里为了通用性，做如下处理：

        save_img = img

        # 情况 A: 如果是 float32 (通常范围是 0.0-1.0 或 真实深度值)
        # 如果你需要无损保存 float，建议用 .npy 或 .exr。但 PNG 必须是整数。
        # 这里假设如果原图是 float，则将其映射到 0-65535。
        # *警告*: 如果你的 TIFF 已经是 uint16，这段逻辑会被跳过，保持原样。
        if img.dtype == np.float32 or img.dtype == np.float64:
            # 归一化并转 uint16
            min_val, max_val = img.min(), img.max()
            if max_val > 0:  # 避免除以0
                # 简单的线性映射
                # 注意：这改变了数值的物理意义，但保留了相对关系（用于可视化或特定训练）
                # 如果你想保持数值绝对值不变（比如深度就是米），通常不能转 PNG，除非乘个系数（比如 *1000 变成毫米）
                # 这里我们假设你是要转格式给 3DGS 用，保持 uint16 精度即可

                # 策略: 如果数值都在 0-1 之间，乘以 65535
                if max_val <= 1.0:
                    save_img = (img * 65535).astype(np.uint16)
                else:
                    # 如果数值很大（比如真实深度），直接由用户决定，这里给个警告
                    # 默认转 uint16
                    save_img = img.astype(np.uint16)

                    # 5. 构建输出路径
        # 替换扩展名为 .png
        root_name = os.path.splitext(filename)[0]
        dst_filename = f"{root_name}.png"
        dst_path = os.path.join(output_dir, dst_filename)

        # 6. 保存 PNG
        # cv2.imwrite 对 uint16 数据会自动保存为 16-bit PNG
        try:
            cv2.imwrite(dst_path, save_img)
            success_count += 1
        except Exception as e:
            print(f"\n[保存失败] {filename}: {e}")
            fail_count += 1

    print("-" * 40)
    print(f"处理完成。成功: {success_count}, 失败: {fail_count}")


def main():
    parser = argparse.ArgumentParser(description="批量将 TIFF 深度/视差图转换为高精度 PNG。")
    parser.add_argument("--input_path", type=str, required=True, help="存放 .tiff 文件的文件夹路径")
    parser.add_argument("--output_path", type=str, required=True, help="保存 .png 文件的文件夹路径")

    args = parser.parse_args()

    convert_tiff_to_png(args.input_path, args.output_path)


if __name__ == "__main__":
    main()