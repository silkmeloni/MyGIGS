import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="将 float32 格式的 .npy 深度图转换为可视化的 RGB 伪彩色图像")

    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="包含 .npy 文件的输入文件夹路径")

    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="保存 RGB 可视化图片的输出文件夹路径")

    parser.add_argument("--max_depth", "-m", type=float, default=None,
                        help="可选：指定全局最大深度值（米）用于归一化。如果不指定，则每张图片单独根据自身的最小值和最大值归一化（对比度最强，但不同图片间的颜色不具备可比性）。对于 Lego 数据集，推荐设置在 4.0 到 6.0 之间以便在序列间保持颜色一致。")

    # 选择 colormap，默认为 TURBO (视觉效果较好)，也可以选 JET 等
    parser.add_argument("--colormap", "-c", type=str, default="TURBO",
                        choices=["TURBO", "JET", "VIRIDIS", "PLASMA", "INFERNO", "MAGMA", "BONE"],
                        help="选择可视化用的色彩映射风格。默认 TURBO。")

    return parser.parse_args()


def get_colormap_enum(cmap_name):
    # 映射字符串到 OpenCV 的常量
    cmaps = {
        "TURBO": cv2.COLORMAP_TURBO,
        "JET": cv2.COLORMAP_JET,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "BONE": cv2.COLORMAP_BONE,  # 黑白
    }
    # 如果 opencv 版本过低没有 TURBO，回退到 JET
    if cmap_name == "TURBO" and not hasattr(cv2, "COLORMAP_TURBO"):
        print("Warning: 当前 OpenCV 版本不支持 COLORMAP_TURBO，回退到 JET。")
        return cv2.COLORMAP_JET
    return cmaps.get(cmap_name, cv2.COLORMAP_JET)


def main():
    args = parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: 输入文件夹不存在 -> {args.input_dir}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    pattern = os.path.join(args.input_dir, "*.npy")
    npy_files = glob.glob(pattern)

    if len(npy_files) == 0:
        print(f"Warning: 在 {args.input_dir} 中没有找到 .npy 文件")
        return

    colormap_enum = get_colormap_enum(args.colormap)

    print(f"正在生成可视化图像 (共 {len(npy_files)} 张)...")
    print(f"样式: {args.colormap}")
    if args.max_depth:
        print(f"归一化模式: 全局固定最大值 ({args.max_depth}m)")
    else:
        print(f"归一化模式: 每张图自动拉伸对比度")

    for npy_path in tqdm(npy_files):
        try:
            # 1. 读取 float32 npy
            depth_map = np.load(npy_path)
            # 处理无效值
            depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0)

            # 2. 归一化到 0-1 之间
            if args.max_depth is not None:
                # 全局归一化 (保持序列间颜色一致)
                # 截断超过 max_depth 的值
                depth_clipped = np.clip(depth_map, 0, args.max_depth)
                depth_norm = depth_clipped / args.max_depth
            else:
                # 单张图自动归一化 (对比度最大化)
                d_min = depth_map.min()
                d_max = depth_map.max()
                if d_max - d_min > 1e-6:
                    depth_norm = (depth_map - d_min) / (d_max - d_min)
                else:
                    # 如果全图深度一样，就全黑
                    depth_norm = np.zeros_like(depth_map)

            # 3. 转换为 0-255 uint8
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # 4. 应用 Colormap (转为 RGB)
            # 注意：OpenCV 的 colormap 通常是低值(近)为冷色/蓝色，高值(远)为暖色/红色。
            # 如果你想反过来（近处暖色），可以在 applyColorMap 之前用 255 - depth_uint8
            depth_color = cv2.applyColorMap(depth_uint8, colormap_enum)

            # 5. 保存
            # r_0.npy -> r_0_viz.png (加个后缀区分)
            file_name = os.path.basename(npy_path).replace(".npy", "_viz.png")
            save_path = os.path.join(args.output_dir, file_name)
            cv2.imwrite(save_path, depth_color)

        except Exception as e:
            print(f"处理失败 {npy_path}: {e}")

    print("可视化图片生成完毕！")


if __name__ == "__main__":
    main()