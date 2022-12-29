import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def main():
    """
    given data that follows the nerfstduio format such as the output from colmap or polycam, convert to a format
    that sdfstudio will ingest
    """

    parser = argparse.ArgumentParser(description="preprocess scannet dataset to sdfstudio dataset")

    parser.add_argument("--data", dest="input_path", help="path to polycam/colmap data directory")
    parser.set_defaults(input_path="NONE")

    parser.add_argument("--output-dir", dest="output_path", help="path to output directory")
    parser.set_defaults(output_path="NONE")

    parser.add_argument("--type", dest="type", required=True, choices=["colmap", "polycam"])
    parser.add_argument("--geo-type", dest="geo_type", default="mono_prior", choices=["mono_prior", "sensor_depth"])
    parser.add_argument("--indoor", action="store_true")

    parser.add_argument("--crop-mult", dest="crop_mult", type=int, default=1)
    parser.add_argument("--omnidata_path", dest="omnidata_path", help="path to omnidata model")
    parser.set_defaults(omnidata_path="/home/yuzh/Projects/omnidata/omnidata_tools/torch")

    parser.add_argument("--pretrained_models", dest="pretrained_models", help="path to pretrained models")
    parser.set_defaults(pretrained_models="/home/yuzh/Projects/omnidata/omnidata_tools/torch/pretrained_models/")

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
        cx = []
        cy = []
        fl_x = []
        fl_y = []

    camera_parameters = camera_parameters["frames"]
    num_frames = len(camera_parameters)

    # load poses
    poses = []
    image_paths = []
    depth_paths = []
    # only load images with corresponding pose info
    # currently in random order??, probably need to sort
    for camera in camera_parameters:
        if POLYCAM:
            # average frames into single intrinsic
            cx.append(camera["cx"])
            cy.append(camera["cy"])
            fl_x.append(camera["fl_x"])
            fl_y.append(camera["fl_y"])

        # OpenGL/Blender convention, needs to change to COLMAP/OpenCV convention
        # https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
        ## IGNORED for now
        c2w = np.array(camera["transform_matrix"]).reshape(4, 4)
        c2w[0:3, 1:3] *= -1

        poses.append(c2w)

        file_path = Path(camera["file_path"])
        img_path = input_path / "images" / file_path.name
        assert img_path.exists()
        image_paths.append(img_path)

        # include sensor depths
        if args.geo_type == "sensor_depth":
            depth_path = input_path / "depths" / f"{file_path.stem}.png"
            assert depth_path.exists()
            depth_paths.append(depth_path)

    poses = np.array(poses)

    if POLYCAM:
        # intrinsics
        camera_intrinsics = []
        for idx in range(len(cx)):
            camera_intrinsics.append(np.array([[fl_x[idx], 0, cx[idx]], [0, fl_y[idx], cy[idx]], [0, 0, 1]]))
    else:
        camera_intrinsics = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

    # camera pose normalization only used for indoor scenes
    if args.indoor:
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
    else:
        scale_mat = np.eye(4).astype(np.float32)

    # copy image
    sample_img = cv2.imread(str(image_paths[0]))
    H, W, _ = sample_img.shape  # 1080 x 1920

    # get smallest side to generate square crop
    target_crop = min(H, W)

    target_size = 384 * args.crop_mult
    trans_totensor = transforms.Compose(
        [
            transforms.CenterCrop(target_crop),
            transforms.Resize(target_size, interpolation=PIL.Image.BILINEAR),
        ]
    )

    depth_trans_totensor = transforms.Compose(
        [
            transforms.Resize([H, W], interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(target_size),
            transforms.Resize(target_size, interpolation=PIL.Image.NEAREST),
        ]
    )

    # center crop by min_dim
    offset_x = (W - target_crop) * 0.5
    offset_y = (H - target_crop) * 0.5

    if POLYCAM:
        for idx in range(len(camera_intrinsics)):
            camera_intrinsics[idx][0, 2] -= offset_x
            camera_intrinsics[idx][1, 2] -= offset_y
            resize_factor = target_size / target_crop
            camera_intrinsics[idx][:2, :] *= resize_factor
    else:
        camera_intrinsics[0, 2] -= offset_x
        camera_intrinsics[1, 2] -= offset_y
        # resize from min_dim x min_dim -> to 384 x 384
        resize_factor = target_size / target_crop
        camera_intrinsics[:2, :] *= resize_factor

    K = camera_intrinsics

    frames = []
    out_index = 0
    for idx, (valid, pose, image_path) in enumerate(tqdm(zip(valid_poses, poses, image_paths))):
        if not valid:
            continue

        # save rgb image
        target_image = output_path / f"{out_index:06d}_rgb.png"
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)

        rgb_path = str(target_image.relative_to(output_path))

        # load depth
        depth_path = depth_paths[idx]
        target_depth_image = output_path / f"{out_index:06d}_sensor_depth.png"
        depth = cv2.imread(str(depth_path), -1).astype(np.float32) / 1000.0

        depth_PIL = Image.fromarray(depth)
        new_depth = depth_trans_totensor(depth_PIL)
        new_depth = np.asarray(new_depth)
        # scale depth as we normalize the scene to unit box
        # new_depth *= scale
        plt.imsave(target_depth_image, new_depth, cmap="viridis")
        np.save(str(target_depth_image).replace(".png", ".npy"), new_depth)

        frame = {
            "rgb_path": rgb_path,
            "camtoworld": pose.tolist(),
            "intrinsics": K[idx].tolist() if isinstance(K, list) else K.tolist(),
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
        "height": target_size,
        "width": target_size,
        "has_mono_prior": args.geo_type == "mono_prior",
        "has_sensor_depth": args.geo_type == "sensor_depth",
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "scene_box": scene_box,
    }

    output_data["frames"] = frames

    # save as json
    with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    if args.geo_type == "mono_prior":
        assert os.path.exists(args.pretrained_models), "Pretrained model path not found"
        assert os.path.exists(args.omnidata_path), "omnidata l path not found"
        # generate mono depth and normal
        print("Generating mono depth")
        os.system(
            f"python scripts/datasets/extract_monocular_cues.py \
            --omnidata_path {args.omnidata_path} \
            --pretrained_model {args.pretrained_models} \
            --img_path {output_path} --output_path {output_path} \
            --task depth"
        )
        print("Generating mono normal")
        os.system(
            f"python scripts/datasets/extract_monocular_cues.py \
            --omnidata_path {args.omnidata_path} \
            --pretrained_model {args.pretrained_models} \
            --img_path {output_path} --output_path {output_path} \
            --task normal"
        )
        print("Done!")


if __name__ == "__main__":
    main()
