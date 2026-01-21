import time
import torch
import viser
import sys
import os
import argparse
import math
import numpy as np
from torch import nn

# ==============================================================================
# ⚠️ TODO: 请在这里修改为你的 GI-GS 项目根目录
# ==============================================================================
PROJECT_PATH = "/path/to/your/GI-GS"  # 修改这里！
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# 尝试导入 GI-GS 的模块
try:
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    # 如果你的 render 参数需要 pipeline，也可以导入
    # from arguments import PipelineParams
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"请检查 PROJECT_PATH 是否正确指向了包含 'scene' 和 'gaussian_renderer' 的目录。")
    exit(1)


# ==============================================================================
# 1. 定义一个 MiniCam 类
# 3DGS 的 render() 函数需要一个 Camera 对象，我们在这里模拟一个
# ==============================================================================
class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear=0.01, zfar=100.0):
        # c2w: camera to world matrix (4x4, torch tensor)
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        # 计算 World-to-Camera (View Matrix)
        # 注意：这里假设 c2w 是标准的图形学矩阵。如果渲染出来是反的，需要给坐标轴乘 -1
        w2c = torch.inverse(c2w).cuda()

        # 3DGS 需要转置的矩阵 (Row-major vs Column-major 的历史遗留问题)
        self.world_view_transform = w2c.transpose(0, 1)

        # 计算投影矩阵 (Projection Matrix)
        self.projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()

        # 组合矩阵 (W2C * Proj)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # 相机中心
        self.camera_center = c2w[:3, 3].cuda()


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# ==============================================================================
# 2. 主逻辑
# ==============================================================================
def main(args):
    # --- 检查路径 ---
    if not os.path.exists(args.ply_path):
        print(f"[Error] 文件不存在: {args.ply_path}")
        return

    # --- 初始化 Viser ---
    print(f"启动 Viewer，访问地址: http://localhost:{args.port}")
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="collapsible")

    # --- 加载模型 ---
    print(f"正在加载 GI-GS 模型 (SH={args.sh_degree})...")
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.ply_path)

    # 将模型数据预加载到 GPU (active_sh_degree 等)
    # 标准 3DGS 不需要额外操作，但某些变体可能需要调用 gaussians.training_setup() 或类似
    # 这里假设 load_ply 已经把 tensor 放到了 CPU/GPU
    print(f"模型加载完毕! 点数: {gaussians.get_xyz.shape[0]}")

    # --- UI 控制面板 ---
    with server.gui.add_folder("PBR Controls"):
        # 材质系数
        gui_roughness = server.gui.add_slider("Roughness Scale", 0.0, 2.0, 0.05, 1.0)
        gui_metallic = server.gui.add_slider("Metallic Scale", 0.0, 2.0, 0.05, 1.0)

        # 光照位置 Handle
        gui_light = server.add_transform_controls("Light Source", position=(0, 2, 0), scale=0.5)

        # 背景颜色
        gui_bg_color = server.gui.add_rgb("Background", (0, 0, 0))

    # --- 渲染回调 (核心) ---
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print("客户端已连接")

        @client.camera.on_update
        def _(cam: viser.CameraHandle):
            # 1. 准备相机参数
            # Viser 给的是 quaternion (wxyz) 和 position (xyz)
            # 我们需要构建 4x4 C2W 矩阵
            c2w = torch.eye(4, dtype=torch.float32)

            # Rotation
            q = torch.tensor(cam.wxyz).unsqueeze(0)
            R = quaternion_to_matrix(q).squeeze(0)  # 需要你自己实现或引用 pytorch3d，或者见下文简单实现
            c2w[:3, :3] = R

            # Position
            c2w[:3, 3] = torch.tensor(cam.position)

            # ⚠️ 坐标系修正：Viser (OpenCV) -> 3DGS (Colmap/OpenGL)
            # 如果画面是倒的，或者旋转方向反了，尝试取消注释下面这行：
            # c2w[:3, 1:3] *= -1

            W = 800  # 渲染分辨率宽
            H = int(W / cam.aspect)

            # 实例化 MiniCam
            viewpoint_cam = MiniCam(
                c2w=c2w,
                width=W,
                height=H,
                fovy=cam.fov,
                fovx=2 * math.atan(math.tan(cam.fov / 2) * cam.aspect)
            )

            # 2. 准备 PBR 参数
            light_pos = torch.tensor(gui_light.position, dtype=torch.float32, device="cuda")
            bg = torch.tensor(gui_bg_color.value, dtype=torch.float32, device="cuda")

            # 3. 调用 GI-GS 渲染器
            # ⚠️ 注意：这里必须匹配你修改后的 render 函数签名
            try:
                render_pkg = render(
                    viewpoint_camera=viewpoint_cam,
                    pc=gaussians,
                    pipe=None,  # 或实例化 PipelineParams()
                    bg_color=bg,
                    # --- 下面是你为了 PBR 新增的参数 ---
                    light_pos=light_pos,
                    roughness_mul=gui_roughness.value,
                    metallic_mul=gui_metallic.value
                )

                # 4. 获取结果并显示
                img_tensor = render_pkg["render"]  # (C, H, W)

                # 转为 numpy (H, W, C) 并发送给 Viser
                img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
                client.set_background_image(img_np)

            except Exception as e:
                print(f"渲染错误 (可能是参数不匹配): {e}")
                # 为了防止控制台刷屏，可以加个 sleep
                time.sleep(1.0)

    while True:
        time.sleep(1.0)


# 辅助函数：四元数转旋转矩阵 (简易版，避免引入 pytorch3d)
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',"--ply_path", type=str, required=True, help="Path to .ply file")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    main(args)