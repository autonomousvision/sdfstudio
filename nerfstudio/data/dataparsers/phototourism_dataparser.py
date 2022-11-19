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
from rich.progress import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.colmap_utils import read_cameras_binary, read_images_binary
from nerfstudio.utils.images import BasicImages

CONSOLE = Console(width=120)


def get_masks(image_idx: int, masks, skys):
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
    return {"mask": mask, "sky": sky}


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

        image_filenames = []
        poses = []

        with CONSOLE.status(f"[bold green]Reading phototourism images and poses for {split} split...") as _:
            cams = read_cameras_binary(self.data / "dense/sparse/cameras.bin")
            imgs = read_images_binary(self.data / "dense/sparse/images.bin")

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

        poses = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_poses=False,
        )

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            camera_type=CameraType.PERSPECTIVE,
        )

        cameras = cameras[indices]
        image_filenames = [image_filenames[i] for i in indices]
        masks = [masks[i] for i in indices]
        skys = [skys[i] for i in indices]

        assert len(cameras) == len(image_filenames)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs={"masks": {"func": get_masks, "kwargs": {"masks": masks, "skys": skys}}},
        )

        return dataparser_outputs
