from PIL import Image
import os


def split_montage(input_path, output_dir="."):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开原始拼图
    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_path}，请检查路径。")
        return

    # 验证尺寸是否符合预期
    width, height = img.size
    print(f"读取到图片尺寸: {width}x{height}")

    sub_width = 800
    sub_height = 800

    # 定义两行四列的文件名
    file_names = [
        ["GT.jpg", "Render.jpg", "Error.jpg", "Depth.jpg"],
        ["Albedo.jpg", "Normal.jpg", "Roughness.jpg", "Metallic.jpg"]
    ]

    # 遍历切图
    for row in range(2):
        for col in range(4):
            # 计算每个子图的边界框 (left, upper, right, lower)
            left = col * sub_width
            upper = row * sub_height
            right = (col + 1) * sub_width
            lower = (row + 1) * sub_height

            # 裁剪并保存
            sub_img = img.crop((left, upper, right, lower))
            output_path = os.path.join(output_dir, file_names[row][col])
            sub_img.save(output_path, quality=95)  # 保持较高画质

            print(f"成功保存: {output_path} (位置: 行{row + 1}, 列{col + 1})")


if __name__ == "__main__":
    # 假设 montage.jpg 和这个脚本在同一个目录下
    input_image = "montage.jpg"

    # 可以在这里指定输出文件夹，比如 output_dir="results"
    split_montage(input_image)