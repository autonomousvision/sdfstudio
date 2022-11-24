# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phototourism dataset parser. Datasets and documentation here: http://phototour.cs.washington.edu/datasets/"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
import yaml
from rich.progress import Console, track
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from nerfstudio.model_components.ray_samplers import save_points
from nerfstudio.utils.images import BasicImages

CONSOLE = Console(width=120)


def get_masks(image_idx: int, masks, skys, sparse_pts):
    """function to process additional mask information

    Args:
        image_idx: specific image index to work with
        mask: mask data
    """

    # mask
    mask = masks[image_idx]
    mask = BasicImages([mask])

    # sky
    sky = skys[image_idx]
    sky = BasicImages([sky])

    # sparse_pts
    pts = sparse_pts[image_idx]
    pts = BasicImages([pts])

    return {"mask": mask, "sky": sky, "sparse_pts": pts}


@dataclass
class PhototourismDataParserConfig(DataParserConfig):
    """Phototourism dataset parser config"""

    _target: Type = field(default_factory=lambda: Phototourism)
    """target class to instantiate"""
    data: Path = Path("data/phototourism/trevi-fountain")
    """Directory specifying location of data."""
    scale_factor: float = 3.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    center_poses: bool = True
    """Whether to center the poses."""


@dataclass
class Phototourism(DataParser):
    """Phototourism dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: PhototourismDataParserConfig

    def __init__(self, config: PhototourismDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.train_split_percentage = config.train_split_percentage

    # pylint: disable=too-many-statements
    def _generate_dataparser_outputs(self, split="train"):

        config_path = self.data / "config.yaml"

        with open(config_path, "r") as yamlfile:
            scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

        sfm_to_gt = np.array(scene_config["sfm2gt"])
        gt_to_sfm = np.linalg.inv(sfm_to_gt)
        sfm_vert1 = gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][0]) + gt_to_sfm[:3, 3]
        sfm_vert2 = gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][1]) + gt_to_sfm[:3, 3]
        bbx_min = np.minimum(sfm_vert1, sfm_vert2)
        bbx_max = np.maximum(sfm_vert1, sfm_vert2)

        image_filenames = []
        poses = []

        with CONSOLE.status(f"[bold green]Reading phototourism images and poses for {split} split...") as _:
            cams = read_cameras_binary(self.data / "dense/sparse/cameras.bin")
            imgs = read_images_binary(self.data / "dense/sparse/images.bin")
            pts3d = read_points3d_binary(self.data / "dense/sparse/points3D.bin")

        # key point depth
        pts3d_array = torch.ones(max(pts3d.keys()) + 1, 4)
        error_array = torch.ones(max(pts3d.keys()) + 1, 1)
        for pts_id, pts in track(pts3d.items(), description="create 3D points", transient=True):
            pts3d_array[pts_id, :3] = torch.from_numpy(pts.xyz)
            error_array[pts_id, 0] = torch.from_numpy(pts.error)

        poses = []
        fxs = []
        fys = []
        cxs = []
        cys = []
        image_filenames = []
        mask_filenames = []
        semantic_filenames = []
        masks = []
        skys = []
        sparse_pts = []

        flip = torch.eye(3)
        flip[0, 0] = -1.0
        flip = flip.double()

        for _id, cam in cams.items():
            img = imgs[_id]

            assert cam.model == "PINHOLE", "Only pinhole (perspective) camera model is supported at the moment"

            pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
            pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
            poses.append(torch.linalg.inv(pose))
            fxs.append(torch.tensor(cam.params[0]))
            fys.append(torch.tensor(cam.params[1]))
            cxs.append(torch.tensor(cam.params[2]))
            cys.append(torch.tensor(cam.params[3]))

            image_filenames.append(self.data / "dense/images" / img.name)
            mask_filenames.append(self.data / "masks" / img.name.replace(".jpg", ".npy"))
            semantic_filenames.append(self.data / "semantic_maps" / img.name.replace(".jpg", ".npz"))

            # load mask
            mask = np.load(mask_filenames[-1])  # ["arr_0"]

            mask = torch.from_numpy(mask).unsqueeze(-1).bool()
            # save nonzeros_indices so we just compute it once
            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
            masks.append(nonzero_indices)

            # load sky
            semantic = np.load(semantic_filenames[-1])["arr_0"]
            is_sky = semantic == 2  # sky id is 2
            skys.append(torch.from_numpy(is_sky).unsqueeze(-1))

            # load sparse 3d points for each view
            # visualize pts3d for each image
            valid_3d_mask = img.point3D_ids != -1
            point3d_ids = img.point3D_ids[valid_3d_mask]
            img_p3d = pts3d_array[point3d_ids]
            img_err = error_array[point3d_ids]
            # img_p3d = img_p3d[img_err[:, 0] < torch.median(img_err)]

            # weight term as in NeuralRecon-W
            err_mean = img_err.mean()
            weight = 2 * np.exp(-((img_err / err_mean) ** 2))

            img_p3d[:, 3:] = weight

            sparse_pts.append(img_p3d)

        poses = torch.stack(poses).float()
        poses[..., 1:3] *= -1
        fxs = torch.stack(fxs).float()
        fys = torch.stack(fys).float()
        cxs = torch.stack(cxs).float()
        cys = torch.stack(cys).float()

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_percentage)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        """
        poses = camera_utils.auto_orient_and_center_poses(
            poses, method=self.config.orientation_method, center_poses=self.config.center_poses
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(poses[:, :3, 3]))

        poses[:, :3, 3] *= scale_factor * self.config.scale_factor
        # shift back so that the object is aligned?
        poses[:, 1, 3] -= 1
        """

        # normalize with scene radius
        radius = scene_config["radius"]
        origin = np.array(scene_config["origin"]).reshape(1, 3)
        origin = torch.from_numpy(origin)
        poses[:, :3, 3] -= origin
        poses[:, :3, 3] *= 1.0 / (radius * 1.01)  # enlarge the radius a little bit

        poses, transform = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_poses=False,
        )

        # scale pts accordingly
        for pts in sparse_pts:
            pts[:, :3] -= origin
            pts[:, :3] *= 1.0 / (radius * 1.01)  # should be the same as pose preprocessing
            pts[:, :3] = pts[:, :3] @ transform[:3, :3].t() + transform[:3, 3:].t()

        # create occupancy grid from sparse points
        points_ori = []
        min_track_length = 10
        for _, p in pts3d.items():
            if p.point2D_idxs.shape[0] > min_track_length:
                points_ori.append(p.xyz)
        points_ori = np.array(points_ori)
        save_points("nori_10.ply", points_ori)

        # filter with bbox
        # normalize cropped area to [-1, -1]
        scene_origin = bbx_min + (bbx_max - bbx_min) / 2

        points_normalized = (points_ori - scene_origin) / (bbx_max - bbx_min)
        # filter out points out of [-1, 1]
        mask = np.prod((points_normalized > -1), axis=-1, dtype=bool) & np.prod(
            (points_normalized < 1), axis=-1, dtype=bool
        )
        points_ori = points_ori[mask]

        save_points("nori_10_filterbbox.ply", points_ori)

        points_ori = torch.from_numpy(points_ori).float()

        # scale pts accordingly
        points_ori -= origin
        points_ori[:, :3] *= 1.0 / (radius * 1.01)  # should be the same as pose preprocessing
        points_ori[:, :3] = points_ori[:, :3] @ transform[:3, :3].t() + transform[:3, 3:].t()

        print(points_ori.shape)

        # expand and quantify

        offset = torch.linspace(-1, 1.0, 3)
        offset_cube = torch.meshgrid(offset, offset, offset)
        offset_cube = torch.stack(offset_cube, dim=-1).reshape(-1, 3)

        voxel_size = 0.25 / (radius * 1.01)
        offset_cube *= voxel_size  # voxel size
        expand_points = points_ori[:, None, :] + offset_cube[None]
        expand_points = expand_points.reshape(-1, 3)
        save_points("expand_points.ply", expand_points.numpy())

        # filter
        # filter out points out of [-1, 1]
        mask = torch.prod((expand_points > -1.0), axis=-1, dtype=torch.bool) & torch.prod(
            (expand_points < 1.0), axis=-1, dtype=torch.bool
        )
        filtered_points = expand_points[mask]
        save_points("filtered_points.ply", filtered_points.numpy())

        grid_size = 32
        voxel_size = 2.0 / grid_size
        quantified_points = torch.floor((filtered_points + 1.0) * grid_size // 2)

        index = quantified_points[:, 0] * grid_size**2 + quantified_points[:, 1] * grid_size + quantified_points[:, 2]

        offset = torch.linspace(-1.0 + voxel_size / 2.0, 1.0 - voxel_size / 2.0, grid_size)
        x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
        offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        # xyz
        mask = torch.zeros(grid_size**3, dtype=torch.bool)
        mask[index.long()] = True

        points_valid = offset_cube[mask]
        save_points("quantified_points.ply", points_valid.numpy())
        # breakpoint()

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        scene_box = mask

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            camera_type=CameraType.PERSPECTIVE,
        )

        # for debug
        # for _ in range(10):
        #    print("==================================================")

        # indices = indices[::20]
        cameras = cameras[indices]
        image_filenames = [image_filenames[i] for i in indices]
        masks = [masks[i] for i in indices]
        skys = [skys[i] for i in indices]
        sparse_pts = [sparse_pts[i] for i in indices]

        assert len(cameras) == len(image_filenames)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs={
                "masks": {"func": get_masks, "kwargs": {"masks": masks, "skys": skys, "sparse_pts": sparse_pts}}
            },
        )

        return dataparser_outputs
