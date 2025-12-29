import os
import json
import argparse
import numpy as np
import shutil
import cv2
from scipy.spatial.transform import Rotation as R  # 这里的 R 是类
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Blender/NeRF-Synthetic dataset to COLMAP format")
    parser.add_argument("--source_path", "-s", type=str, required=True,
                        help="Path to Blender dataset (containing transforms_train.json)")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to save COLMAP formatted data")
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Number of random points to generate in points3D.txt")
    parser.add_argument("--white_background", action="store_true",
                        help="Convert transparent background to white (default is black if not set)")
    return parser.parse_args()


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


# === 修改点在这里 ===
def rotmat2qvec(rotmat):  # 参数名从 R 改为 rotmat
    # 使用全局的 R (Rotation 类) 来处理传入的 rotmat 矩阵
    r = R.from_matrix(rotmat)
    # scipy returns (x, y, z, w), COLMAP needs (w, x, y, z)
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def process_image(src_path, dst_path, white_bg=False):
    """读取图片，处理透明背景，保存"""
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Cannot read {src_path}")
        return

    # Handle Alpha channel
    if len(img.shape) > 2 and img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        rgb = img[:, :, :3]

        bg_color = 255 if white_bg else 0

        # Blend
        new_img = (rgb * alpha[..., None] + bg_color * (1 - alpha[..., None])).astype(np.uint8)
    else:
        new_img = img

    cv2.imwrite(dst_path, new_img)


def main():
    args = parse_args()

    # Setup directories
    images_out_dir = os.path.join(args.output_path, "images")
    sparse_out_dir = os.path.join(args.output_path, "sparse/0")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(sparse_out_dir, exist_ok=True)

    # Loads JSONs
    splits = ["train", "test"]  # usually val is not used for colmap reconstruction
    frames = []

    print("Loading JSONs...")
    camera_angle_x = None

    for split in splits:
        json_path = os.path.join(args.source_path, f"transforms_{split}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                meta = json.load(f)
                if camera_angle_x is None:
                    camera_angle_x = meta.get("camera_angle_x")

                # Append split info to frames to handle filenames
                for frame in meta["frames"]:
                    frame["split"] = split
                    frames.append(frame)

    if not frames:
        print("Error: No frames found in transforms_*.json")
        return

    # --- 1. Write cameras.txt ---
    print("Processing Cameras...")
    # Get image size from the first valid image
    first_img_path = os.path.join(args.source_path, frames[0]["file_path"])
    # Handle missing extension in json
    if not os.path.exists(first_img_path) and not first_img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        first_img_path += ".png"

    tmp_img = cv2.imread(first_img_path)
    if tmp_img is None:
        print(f"Error: Could not read first image to determine resolution: {first_img_path}")
        return
    H, W = tmp_img.shape[:2]

    # Calculate Focal Length
    # f = 0.5 * W / tan(0.5 * angle_x)
    fl_x = 0.5 * W / np.tan(0.5 * camera_angle_x)
    fl_y = fl_x  # Assume square pixels
    cx = W / 2.0
    cy = H / 2.0

    with open(os.path.join(sparse_out_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera.\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # ID 1, PINHOLE, W, H, fx, fy, cx, cy
        f.write(f"1 PINHOLE {W} {H} {fl_x} {fl_y} {cx} {cy}\n")

    # --- 2. Write images.txt & Copy Images ---
    print("Processing Images and Extrinsics...")

    # Coordinate conversion matrix (OpenGL to OpenCV)
    # OpenGL: Right, Up, Back (-Z view)
    # OpenCV: Right, Down, Forward (+Z view)
    # Needs to flip Y and Z
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    points_center = np.zeros(3)

    with open(os.path.join(sparse_out_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image.\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, frame in enumerate(tqdm(frames)):
            img_id = idx + 1

            # 1. Image Path Handling
            original_path = frame["file_path"]
            # Fix path if it doesn't have extension
            if not original_path.lower().endswith(('.png', '.jpg')):
                original_path += ".png"

            # Remove ./ if present
            clean_path = original_path.replace("./", "")
            full_src_path = os.path.join(args.source_path, clean_path)

            # Create unique filename for flattened output
            # e.g. train_r_0.png
            base_name = os.path.basename(clean_path)
            new_name = f"{frame['split']}_{base_name}"
            full_dst_path = os.path.join(images_out_dir, new_name)

            # Process and Copy Image
            process_image(full_src_path, full_dst_path, args.white_background)

            # 2. Extrinsics Processing
            # Blender c2w
            c2w = np.array(frame["transform_matrix"])

            # Convert coordinate system (OpenGL -> OpenCV)
            c2w = np.matmul(c2w, flip_mat)

            # COLMAP needs w2c (World to Camera)
            w2c = np.linalg.inv(c2w)

            R_mat = w2c[:3, :3]
            T_vec = w2c[:3, 3]

            # Rotation Matrix to Quaternion (w, x, y, z)
            qvec = rotmat2qvec(R_mat)

            # Accumulate center for point cloud generation
            points_center += c2w[:3, 3]

            # Write to file
            f.write(f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T_vec[0]} {T_vec[1]} {T_vec[2]} 1 {new_name}\n")
            f.write("\n")  # Empty 2D points line

    # --- 3. Write points3D.txt (Random Initialization) ---
    print("Generating Fake Point Cloud...")
    if len(frames) > 0:
        points_center /= len(frames)

    with open(os.path.join(sparse_out_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point.\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        # Generate random points around the scene center
        # Blender scenes are usually normalized within [-1.5, 1.5] roughly
        rng = np.random.default_rng(42)
        xyzs = (rng.random((args.num_points, 3)) - 0.5) * 3.0

        rgbs = rng.integers(0, 255, (args.num_points, 3))

        for i in range(args.num_points):
            pt_id = i + 1
            f.write(
                f"{pt_id} {xyzs[i, 0]:.6f} {xyzs[i, 1]:.6f} {xyzs[i, 2]:.6f} {rgbs[i, 0]} {rgbs[i, 1]} {rgbs[i, 2]} 0.0\n")

    print(f"Done! COLMAP data saved to {args.output_path}")
    print(f"  - Images: {len(frames)}")
    print(f"  - Points: {args.num_points}")


if __name__ == "__main__":
    main()