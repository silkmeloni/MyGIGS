import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def get_omnidata_normal_model(repo_path=None):
    """
    加载 Omnidata 法线估计模型
    :param repo_path: 本地 Omnidata 仓库路径 (如果为 None，则从 GitHub 下载)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if repo_path and os.path.exists(repo_path):
        print(f"Loading Omnidata model from local path: {repo_path}")
        # source='local' 表示从本地加载代码
        # 注意：pretrained=True 依然会尝试去 torch cache 找权重，找不到才会下载
        # 请确保权重文件 omnidata_dpt_normal_v2.ckpt 已放入 ~/.cache/torch/hub/checkpoints/
        model = torch.hub.load(repo_path, 'model', 'normal', pretrained=True)
    else:
        print("Loading Omnidata model from GitHub (requires internet)...")
        try:
            model = torch.hub.load('EPFL-VILAB/omnidata', 'model', 'normal', pretrained=True)
        except Exception as e:
            print(f"\n[Error] Failed to load from GitHub: {e}")
            print("Tip: If network is bad, clone the repo locally and use --omnidata_path")
            raise e

    model.to(device)
    model.eval()
    return model


def predict_normal(model, image_path, output_path=None):
    device = next(model.parameters()).device

    # 1. 读取图像
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)

    # 2. 预处理
    # Omnidata 推荐输入尺寸为 384x384 或其倍数
    target_size = 384

    # 注意：如果图像是非正方形，CenterCrop 会丢失信息。
    # 更好的做法是 Resize 到 (384, 384) 进行推理，然后再 Resize 回去。
    trans_resize_only = transforms.Compose([
        transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = trans_resize_only(img).unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        # 输出是 [1, 3, H, W]，范围在 [0, 1]
        pred_normal = model(img_tensor)

        # 4. 后处理
        # 截断到 [0, 1]
        pred_normal = torch.clamp(pred_normal, 0, 1)

        # 重新映射到 [-1, 1] 用于可视化或保存 (可选，取决于你的后续用途)
        # 3DGS 通常需要 [0, 1] 存储为图片，或者 [-1, 1] 作为数据
        # 这里我们先恢复到原始尺寸
        pred_normal = F.interpolate(pred_normal, size=(original_size[1], original_size[0]), mode='bilinear',
                                    align_corners=False)

        # 转为 numpy image [H, W, 3]
        normal_img = pred_normal.squeeze().permute(1, 2, 0).cpu().numpy()

    # 5. 保存结果
    if output_path:
        # 保存为 [0, 255] 的 PNG 图片
        plt.imsave(output_path, normal_img)
        print(f"Normal map saved to {output_path}")

    return normal_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omnidata Normal Estimation")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, default="normal_omnidata.png", help="Path to save output normal map")
    parser.add_argument("--omnidata_path", type=str, default=None,
                        help="Path to local cloned omnidata repository (e.g., './omnidata')")
    args = parser.parse_args()

    model = get_omnidata_normal_model(repo_path=args.omnidata_path)
    predict_normal(model, args.img_path, args.output_path)