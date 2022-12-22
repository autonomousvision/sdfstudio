import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms


def main():
    """
    given data that follows the nerfstduio format such as the output from colmap or polycam, convert to a format
    that sdfstudio will ingest
    """

    parser = argparse.ArgumentParser(description="preprocess scannet dataset to sdfstudio dataset")

    parser.add_argument("--data", dest="input_path", help="path to scannet scene")
    parser.set_defaults(input_path="NONE")

    parser.add_argument("--output-dir", dest="output_path", help="path to output")
    parser.set_defaults(output_path="NONE")

    parser.add_argument("--type", dest="type", default="colmap", choices=["colmap", "polycam"])

    args = parser.parse_args()

    output_path = Path(args.output_path)
    input_path = Path(args.input_path)

    POLYCAM = True if args.type == "polycam" else False

    output_path.mkdir(parents=True, exist_ok=True)

    # load transformation json with images/intrinsics/extrinsics
    camera_parameters_path = input_path / "transforms.json"
    camera_parameters = json.load(open(camera_parameters_path))

    # extract intrinsic parameters
    if not POLYCAM:
        cx = camera_parameters["cx"]
        cy = camera_parameters["cy"]
        fl_x = camera_parameters["fl_x"]
        fl_y = camera_parameters["fl_y"]
    else:
        cx = 0
        cy = 0
        fl_x = 0
        fl_y = 0

    camera_parameters = camera_parameters["frames"]
    num_frames = len(camera_parameters)

    camera_intrinsic = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

    # load poses
    poses = []
    image_paths = []
    # only load images with corresponding pose info
    # currently in random order??, probably need to sort
    for camera in camera_parameters:
        if POLYCAM:
            # average frames into single intrinsic
            cx += camera["cx"]
            cy += camera["cy"]
            fl_x += camera["fl_x"]
            fl_y += camera["fl_y"]

        # OpenGL/Blender convention, needs to change to COLMAP/OpenCV convention
        # https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
        ## IGNORED for now
        c2w = np.array(camera["transform_matrix"]).reshape(4, 4)
        c2w[0:3, 1:3] *= -1

        img_path = input_path / camera["file_path"]
        assert img_path.exists()
        image_paths.append(img_path)
        poses.append(c2w)

    poses = np.array(poses)

    if POLYCAM:
        # intrinsics
        cx /= num_frames
        cy /= num_frames
        fl_x /= num_frames
        fl_y /= num_frames

    camera_intrinsic = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

    center = (min_vertices + max_vertices) / 2.0
    scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)

    # we should normalize pose to unit cube
    poses[:, :3, 3] -= center
    poses[:, :3, 3] *= scale

    # inverse normalization
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] -= center
    scale_mat[:3] *= scale
    scale_mat = np.linalg.inv(scale_mat)

    # copy image
    sample_img = cv2.imread(str(image_paths[0]))
    H, W, _ = sample_img.shape  # 1080 x 1920

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

    K = camera_intrinsic

    frames = []
    out_index = 0
    for idx, (valid, pose, image_path) in enumerate(zip(valid_poses, poses, image_paths)):
        if not valid:
            continue

        target_image = output_path / f"{out_index:06d}_rgb.png"
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)

        rgb_path = str(target_image.relative_to(output_path))
        frame = {
            "rgb_path": rgb_path,
            "camtoworld": pose.tolist(),
            "intrinsics": K.tolist(),
            "mono_depth_path": rgb_path.replace("_rgb.png", "_depth.npy"),
            "mono_normal_path": rgb_path.replace("_rgb.png", "_normal.npy"),
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
        "height": target_size,
        "width": target_size,
        "has_mono_prior": True,
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "scene_box": scene_box,
    }

    output_data["frames"] = frames

    # save as json
    with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
