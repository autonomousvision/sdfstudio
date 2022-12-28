import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description="preprocess scannet dataset to sdfstudio dataset")

parser.add_argument("--input_path", dest="input_path", help="path to scannet scene")
parser.set_defaults(im_name="NONE")

parser.add_argument("--output_path", dest="output_path", help="path to output")
parser.set_defaults(store_name="NONE")
parser.add_argument(
    "--type",
    dest="type",
    default="mono_prior",
    choices=["mono_prior", "sensor_depth"],
    help="mono_prior to use monocular prior, sensor_depth to use depth captured with a depth sensor (gt depth)",
)

args = parser.parse_args()

image_size = 384
trans_totensor = transforms.Compose(
    [
        transforms.CenterCrop(image_size * 2),
        transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
    ]
)

depth_trans_totensor = transforms.Compose(
    [
        transforms.Resize([968, 1296], interpolation=PIL.Image.NEAREST),
        transforms.CenterCrop(image_size * 2),
        transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
    ]
)

output_path = Path(args.output_path)  # "data/custom/scannet_scene0050_00"
input_path = Path(args.input_path)  # "/home/yuzh/Projects/datasets/scannet/scene0050_00"

output_path.mkdir(parents=True, exist_ok=True)

# load color
color_path = input_path / "frames" / "color"
color_paths = sorted(glob.glob(os.path.join(color_path, "*.jpg")), key=lambda x: int(os.path.basename(x)[:-4]))

# load depth
depth_path = input_path / "frames" / "depth"
depth_paths = sorted(glob.glob(os.path.join(depth_path, "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))


# load intrinsic
intrinsic_path = input_path / "frames" / "intrinsic" / "intrinsic_color.txt"
camera_intrinsic = np.loadtxt(intrinsic_path)

# load pose
pose_path = input_path / "frames" / "pose"
poses = []
pose_paths = sorted(glob.glob(os.path.join(pose_path, "*.txt")), key=lambda x: int(os.path.basename(x)[:-4]))
for pose_path in pose_paths:
    c2w = np.loadtxt(pose_path)
    poses.append(c2w)
poses = np.array(poses)

# deal with invalid poses
valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
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

# copy image
H, W = 968, 1296

# center crop by 2 * image_size
offset_x = (W - image_size * 2) * 0.5
offset_y = (H - image_size * 2) * 0.5
camera_intrinsic[0, 2] -= offset_x
camera_intrinsic[1, 2] -= offset_y
# resize from 384*2 to 384
resize_factor = 0.5
camera_intrinsic[:2, :] *= resize_factor

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
    img = Image.open(image_path)
    img_tensor = trans_totensor(img)
    img_tensor.save(target_image)

    # load depth
    target_depth_image = output_path / f"{out_index:06d}_sensor_depth.png"
    depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0

    depth_PIL = Image.fromarray(depth)
    new_depth = depth_trans_totensor(depth_PIL)
    new_depth = np.asarray(new_depth)
    # scale depth as we normalize the scene to unit box
    new_depth *= scale
    plt.imsave(target_depth_image, new_depth, cmap="viridis")
    np.save(str(target_depth_image).replace(".png", ".npy"), new_depth)

    rgb_path = str(target_image.relative_to(output_path))
    frame = {
        "rgb_path": rgb_path,
        "camtoworld": pose.tolist(),
        "intrinsics": K.tolist(),
        "mono_depth_path": rgb_path.replace("_rgb.png", "_depth.npy"),
        "mono_normal_path": rgb_path.replace("_rgb.png", "_normal.npy"),
        "sensor_depth_path": rgb_path.replace("_rgb.png", "_sensor_depth.npy"),
    }

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
    "height": image_size,
    "width": image_size,
    "has_mono_prior": True,
    "has_sensor_depth": True,
    "pairs": None,
    "worldtogt": scale_mat.tolist(),
    "scene_box": scene_box,
}

output_data["frames"] = frames

# save as json
with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)
