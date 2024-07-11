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

"""Data parser for friends dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing_extensions import Dict, Literal, Optional, Type

import cv2
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchtyping import TensorType

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

# random rotate
# from scipy.spatial.transform import Rotation
# random_rotation = torch.eye(4)
# random_rotation[:3, :3] = torch.from_numpy(Rotation.random().as_matrix())


CONSOLE = Console()


def get_src_from_pairs(
    ref_idx, all_imgs, pairs_srcs, neighbors_num=None, neighbors_shuffle=False
) -> Dict[str, TensorType]:
    # src_idx[0] is ref img
    src_idx = pairs_srcs[ref_idx]
    # randomly sample neighbors
    if neighbors_num and neighbors_num > -1 and neighbors_num < len(src_idx) - 1:
        if neighbors_shuffle:
            perm_idx = torch.randperm(len(src_idx) - 1) + 1
            src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_num]]])
        else:
            src_idx = src_idx[: neighbors_num + 1]
    src_idx = src_idx.to(all_imgs.device)
    return {"src_imgs": all_imgs[src_idx], "src_idxs": src_idx}


def get_image(image_filename, alpha_color=None) -> TensorType["image_height", "image_width", "num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_depths_and_normals(image_idx: int, depths, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    depth = depths[image_idx]
    # normal
    normal = normals[image_idx]

    return {"depth": depth, "normal": normal}


@dataclass
class MonoSDFDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: MonoSDFScene)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    include_mono_prior: bool = False
    """whether or not to include loading of normal """
    downscale_factor: int = 1
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Friends axis-aligned bbox will be scaled to this value.
    """
    center_crop_type: Literal[
        "center_crop_for_replica", "center_crop_for_tnt", "center_crop_for_dtu", "no_crop"
    ] = "center_crop_for_dtu"
    """center crop type as monosdf, we should create a dataset that don't need this"""
    load_pairs: bool = False
    """whether to load pairs for multi-view consistency"""
    neighbors_num: Optional[int] = None
    neighbors_shuffle: Optional[bool] = False
    pairs_sorted_ascending: Optional[bool] = True
    """if src image pairs are sorted in ascending order by similarity i.e. the last element is the most similar to the first (ref)"""


@dataclass
class MonoSDFScene(DataParser):
    """UniScene Dataset"""

    config: MonoSDFDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths

        image_paths = glob_data(str(self.config.data / "*_rgb.png"))

        depth_paths = glob_data(str(self.config.data / "*_depth.npy"))
        normal_paths = glob_data(str(self.config.data / "*_normal.npy"))

        n_images = len(image_paths)

        cam_file = self.config.data / "cameras.npz"
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict[f"scale_mat_{idx}"].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict[f"world_mat_{idx}"].astype(np.float32) for idx in range(n_images)]

        intrinsics_all = []
        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)

            # chair
            # scale = 384 / 1080
            # offset = (1440 - 1080) * 0.5
            # thin
            # scale = 384 / 1020
            # offset = (1360 - 1020) * 0.5

            # intrinsics[0, 2] -= offset
            # intrinsics[:2, :] *= scale

            center_crop_type = self.config.center_crop_type
            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == "center_crop_for_replica":
                scale = 384 / 680
                offset = (1200 - 680) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == "center_crop_for_tnt":
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == "center_crop_for_dtu":
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == "padded_for_dtu":
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif (
                center_crop_type == "no_crop"
            ):  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError

            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())

        image_filenames = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for idx in range(n_images):
            # unpack data
            image_filename = image_paths[idx]
            # TODO now we has the first intrincis
            intrinsics = intrinsics_all[idx]
            camtoworld = pose_all[idx]
            # append data
            image_filenames.append(image_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)
        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        if self.config.include_mono_prior:
            # load monocular depths and normals
            depth_images, normal_images = [], []
            for idx, (dpath, npath) in enumerate(zip(depth_paths, normal_paths)):
                depth = np.load(dpath)
                depth_images.append(torch.from_numpy(depth).float())

                normal = np.load(npath)

                # important as the output of omnidata is normalized
                normal = normal * 2.0 - 1.0
                normal = torch.from_numpy(normal).float()

                # transform normal to world coordinate system
                rot = camera_to_worlds[idx][:3, :3].clone()

                normal_map = normal.reshape(3, -1)
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

                normal_map = rot @ normal_map
                normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
                normal_images.append(normal_map)

            # stack
            depth_images = torch.stack(depth_images)
            normal_images = torch.stack(normal_images)
        else:
            depth_images = None
            normal_images = None

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        # camera_to_worlds = random_rotation[None] @ camera_to_worlds

        scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32))

        height, width = get_image(image_filenames[0]).shape[:2]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO fix later
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)
        if self.config.include_mono_prior:
            additional_inputs_dict = {
                "cues": {"func": get_depths_and_normals, "kwargs": {"depths": depth_images, "normals": normal_images}}
            }
        else:
            additional_inputs_dict = {}

        pairs_path = self.config.data / "pairs.txt"
        if pairs_path.exists() and split == "train" and self.config.load_pairs:
            with open(pairs_path, "r") as f:
                pairs = f.readlines()
            split_ext = lambda x: x.split(".")[0]
            pairs_srcs = []
            for sources_line in pairs:
                sources_array = [int(split_ext(img_name)) for img_name in sources_line.split(" ")]
                if self.config.pairs_sorted_ascending:
                    # invert (flip) the source elements s.t. the most similar source is in index 1 (index 0 is reference)
                    sources_array = [sources_array[0]] + sources_array[:1:-1]
                pairs_srcs.append(sources_array)
            pairs_srcs = torch.tensor(pairs_srcs)
            all_imgs = torch.stack(
                [get_image(image_filename) for image_filename in sorted(image_filenames)], axis=0
            ).cuda()

            additional_inputs_dict["pairs"] = {
                "func": get_src_from_pairs,
                "kwargs": {
                    "all_imgs": all_imgs,
                    "pairs_srcs": pairs_srcs,
                    "neighbors_num": self.config.neighbors_num,
                    "neighbors_shuffle": self.config.neighbors_shuffle,
                },
            }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            depths=depth_images,
            normals=normal_images,
        )
        return dataparser_outputs
