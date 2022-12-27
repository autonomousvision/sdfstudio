import argparse
import glob
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description="preprocess neural rgbd dataset to sdfstudio dataset")

parser.add_argument("--input_path", dest="input_path", help="path to scannet scene")
parser.add_argument("--output_path", dest="output_path", help="path to output")
parser.add_argument(
    "--type",
    dest="type",
    default="mono_prior",
    choices=["mono_prior", "sensor_depth"],
    help="mono_prior to use monocular prior, sensor_depth to use depth captured with a depth sensor (gt depth)",
)

args = parser.parse_args()


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split("([0-9]+)", s)]


def load_poses(posefile):
    file = open(posefile, "r")
    lines = file.readlines()
    file.close()
    poses = []
    valid = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if "nan" in lines[i]:
            valid.append(False)
            poses.append(np.eye(4, 4, dtype=np.float32).tolist())
        else:
            valid.append(True)
            pose_floats = [[float(x) for x in line.split()] for line in lines[i : i + lines_per_matrix]]
            poses.append(pose_floats)

    return poses, valid


output_path = Path(args.output_path)  # "data/neural_rgbd/breakfast_room/"
input_path = Path(args.input_path)  # "data/neural_rgbd_data/breakfast_room/"

output_path.mkdir(parents=True, exist_ok=True)

# load color
color_path = input_path / "images"
color_paths = sorted(glob.glob(os.path.join(color_path, "*.png")), key=alphanum_key)

# load depth
depth_path = input_path / "depth_filtered"
depth_paths = sorted(glob.glob(os.path.join(depth_path, "*.png")), key=alphanum_key)

H, W = cv2.imread(depth_paths[0]).shape[:2]
print(H, W)

# load intrinsic
intrinsic_path = input_path / "focal.txt"
focal_length = np.loadtxt(intrinsic_path)

camera_intrinsic = np.eye(4)
camera_intrinsic[0, 0] = focal_length
camera_intrinsic[1, 1] = focal_length
camera_intrinsic[0, 2] = W * 0.5
camera_intrinsic[1, 2] = H * 0.5

print(camera_intrinsic)
# load pose

pose_path = input_path / "poses.txt"
poses, valid_poses = load_poses(pose_path)
poses = np.array(poses)
print(poses.shape)

# OpenGL/Blender convention, needs to change to COLMAP/OpenCV convention
# https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
poses[:, 0:3, 1:3] *= -1

# deal with invalid poses
min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

center = (min_vertices + max_vertices) / 2.0
scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
print(center, scale)

# we should normalize pose to unit cube
poses[:, :3, 3] -= center
poses[:, :3, 3] *= scale

# inverse normalization
scale_mat = np.eye(4).astype(np.float32)
scale_mat[:3, 3] -= center
scale_mat[:3] *= scale
scale_mat = np.linalg.inv(scale_mat)

if args.type == "mono_prior":
    # center copy image if use monocular prior because omnidata use 384x384 as inputs
    # get smallest side to generate square crop
    target_crop = min(H, W)

    target_size = 384
    trans_totensor = transforms.Compose(
        [
            transforms.CenterCrop(target_crop),
            transforms.Resize(target_size, interpolation=PIL.Image.BILINEAR),
        ]
    )

    # center crop by min_dim
    offset_x = (W - target_crop) * 0.5
    offset_y = (H - target_crop) * 0.5

    camera_intrinsic[0, 2] -= offset_x
    camera_intrinsic[1, 2] -= offset_y
    # resize from min_dim x min_dim -> to 384 x 384
    resize_factor = target_size / target_crop
    camera_intrinsic[:2, :] *= resize_factor

    # new H, W after center crop
    H, W = target_size, target_size

K = camera_intrinsic

frames = []
out_index = 0
for idx, (valid, pose, image_path, depth_path) in enumerate(zip(valid_poses, poses, color_paths, depth_paths)):

    if idx % 10 != 0:
        continue
    if not valid:
        continue

    target_image = output_path / f"{out_index:06d}_rgb.png"
    print(target_image)
    if args.type == "mono_prior":
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)
    else:
        shutil.copyfile(image_path, target_image)

    rgb_path = str(target_image.relative_to(output_path))
    frame = {
        "rgb_path": rgb_path,
        "camtoworld": pose.tolist(),
        "intrinsics": K.tolist(),
    }
    if args.type == "mono_prior":
        frame.update(
            {
                "mono_depth_path": rgb_path.replace("_rgb.png", "_depth.npy"),
                "mono_normal_path": rgb_path.replace("_rgb.png", "_normal.npy"),
            }
        )
    else:
        frame["sensor_depth_path"] = rgb_path.replace("_rgb.png", "_depth.npy")

        depth_map = cv2.imread(depth_path, -1)
        # Convert depth to meters, then to "network units"
        depth_shift = 1000.0
        depth_maps = (np.array(depth_map) / depth_shift).astype(np.float32)
        depth_maps *= scale

        np.save(output_path / frame["sensor_depth_path"], depth_maps)

        # color map gt depth for visualization
        plt.imsave(output_path / frame["sensor_depth_path"].replace(".npy", ".png"), depth_maps, cmap="viridis")

    frames.append(frame)
    out_index += 1

# scene bbox for the scannet scene
scene_box = {
    "aabb": [[-1, -1, -1], [1, 1, 1]],
    "near": 0.05,
    "far": 2.5,
    "radius": 1.0,
    "collider_type": "box",
}

# meta data
output_data = {
    "camera_model": "OPENCV",
    "height": H,
    "width": W,
    "has_mono_prior": args.type == "mono_prior",
    "has_sensor_depth": args.type == "sensor_depth",
    "pairs": None,
    "worldtogt": scale_mat.tolist(),
    "scene_box": scene_box,
}

output_data["frames"] = frames

# save as json
with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)
