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

"""
Implementation of patch warping for multi-view consistency loss.
"""
import math
from typing import Optional, Union

import torch
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal
import numpy as np

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.model_components.ray_samplers import save_points


def get_intersection_points(ray_samples: RaySamples, sdf: torch.Tensor, normal: torch.Tensor):
    """compute intersection points

    Args:
        ray_samples (RaySamples): _description_
        sdf (torch.Tensor): _description_
        normal (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    # TODO we should support different ways to compute intersections

    # Calculate if sign change occurred and concat 1 (no sign change) in
    # last dimension
    n_rays, n_samples = ray_samples.shape
    starts = ray_samples.frustums.starts
    sign_matrix = torch.cat([torch.sign(sdf[:, :-1, 0] * sdf[:, 1:, 0]), torch.ones(n_rays, 1).to(sdf.device)], dim=-1)
    cost_matrix = sign_matrix * torch.arange(n_samples, 0, -1).float().to(sdf.device)

    # Get first sign change and mask for values where a.) a sign changed
    # occurred and b.) no a neg to pos sign change occurred (meaning from
    # inside surface to outside)
    values, indices = torch.min(cost_matrix, -1)
    mask_sign_change = values < 0
    mask_pos_to_neg = sdf[torch.arange(n_rays), indices, 0] > 0

    # Define mask where a valid depth value is found
    mask = mask_sign_change & mask_pos_to_neg

    # Get depth values and function values for the interval
    d_low = starts[torch.arange(n_rays), indices, 0][mask]
    v_low = sdf[torch.arange(n_rays), indices, 0][mask]
    n_low = normal[torch.arange(n_rays), indices, :][mask]

    indices = torch.clamp(indices + 1, max=n_samples - 1)
    d_high = starts[torch.arange(n_rays), indices, 0][mask]
    v_high = sdf[torch.arange(n_rays), indices, 0][mask]
    n_high = normal[torch.arange(n_rays), indices, :][mask]

    # linear-interpolations or run secant method to refine depth
    z = (v_low * d_high - v_high * d_low) / (v_low - v_high)

    # make this simpler
    origins = ray_samples.frustums.origins[torch.arange(n_rays), indices, :][mask]
    directions = ray_samples.frustums.directions[torch.arange(n_rays), indices, :][mask]

    intersection_points = origins + directions * z[..., None]

    # interpolate normal for simplicity so we don't need to call the model again
    points_normal = (v_low[..., None] * n_high - v_high[..., None] * n_low) / (v_low[..., None] - v_high[..., None])

    points_normal = torch.nn.functional.normalize(points_normal, dim=-1, p=2)
    return intersection_points, points_normal, mask


def get_homography(intersection_points: torch.Tensor, normal: torch.Tensor, cameras: Cameras):
    """get homography

    Args:
        intersection_points (torch.Tensor): _description_
        normal (torch.Tensor): _description_
        cameras (Cameras): _description_
    """
    device = intersection_points.device

    # construct homography
    c2w = cameras.camera_to_worlds.to(device)
    K = cameras.get_intrinsics_matrices().to(device)
    K_inv = torch.linalg.inv(K)

    # convert camera to opencv format
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, np.array([1, 0, 2]), :]
    c2w[:, 2, :] *= -1

    T1 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
    T1 = torch.from_numpy(T1).to(device)

    T2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    T2 = torch.from_numpy(T2).to(device)

    T3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
    T3 = torch.from_numpy(T3).to(device)

    T = T3 @ T2
    T_inv = torch.linalg.inv(T)

    # also for points and normal
    intersection_points = intersection_points @ T_inv[:3, :3]
    normal = normal @ T_inv[:3, :3]
    # end convert

    w2c_r = c2w[:, :3, :3].transpose(1, 2)
    w2c_t = -w2c_r @ c2w[:, :3, 3:]
    w2c = torch.cat([w2c_r, w2c_t], dim=-1)

    R_rel = w2c[:, :3, :3] @ c2w[:1, :3, :3]  # [N, 3, 3]
    t_rel = w2c[:, :3, :3] @ c2w[:1, :3, 3:] + w2c[:1, :3, 3:]  # [N, 3, 1]

    p_ref = w2c[0, :3, :3] @ intersection_points.transpose(1, 0) + w2c[0, :3, 3:]  # [3, n_pts]
    n_ref = w2c[0, :3, :3] @ normal.transpose(1, 0)  # [3, n_pts]

    d = torch.sum(n_ref * p_ref, dim=0, keepdims=True)
    # TODO make this clear
    H = R_rel[:, None, :, :] + t_rel[:, None, :, :] @ n_ref.transpose(1, 0)[None, :, None, :] / d[..., None, None]

    H = K[:, None] @ H @ K_inv[None, :1]  # [n_cameras, n_pts, 3, 3]
    return H


class PatchWarping(nn.Module):
    """Standard patch warping."""

    def __init__(self, patch_size: int = 31, pixel_offset: float = 0.5):
        super().__init__()

        self.patch_size = patch_size
        half_size = patch_size // 2

        # generate pattern
        patch_coords = torch.meshgrid(
            torch.arange(-half_size, half_size + 1), torch.arange(-half_size, half_size + 1), indexing="xy"
        )

        patch_coords = torch.stack(patch_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        self.patch_coords = torch.cat([patch_coords, torch.zeros_like(patch_coords[..., :1])], dim=-1)

    def forward(
        self,
        ray_samples: RaySamples,
        sdf: torch.Tensor,
        normal: torch.Tensor,
        cameras: Cameras,
        images: torch.Tensor,
        pix_indices: torch.Tensor,
    ):

        device = sdf.device

        # find intersection points and normals
        intersection_points, normal, mask = get_intersection_points(ray_samples, sdf, normal)

        H = get_homography(intersection_points, normal, cameras)

        # Attention uv is (y, x) and we should change to (x, y) for homography
        pix_indices = torch.flip(pix_indices, dims=[-1])[mask].float()
        pix_indices = torch.cat([pix_indices, torch.ones(pix_indices.shape[0], 1).to(device)], dim=-1)  # [n_pts, 3]

        pix_indices = pix_indices[:, None, None, :] + self.patch_coords[None].to(device)  # [n_pts, patch_h, patch_w, 3]
        pix_indices = pix_indices.permute(0, 3, 1, 2).reshape(
            1, -1, 3, self.patch_size**2
        )  # [1, n_pts, 3, patch_h*patch_w]

        warped_indices = H @ pix_indices
        warped_indices = warped_indices[:, :, :2, :] / warped_indices[:, :, 2:, :]

        pix_coords = warped_indices.permute(0, 1, 3, 2).contiguous()  # [..., :2]
        pix_coords[..., 0] /= cameras.image_width[:, None, None].to(device) - 1
        pix_coords[..., 1] /= cameras.image_height[:, None, None].to(device) - 1
        pix_coords = (pix_coords - 0.5) * 2

        # valid
        valid = (
            (pix_coords[..., 0] > -1.0)
            & (pix_coords[..., 0] < 1.0)
            & (pix_coords[..., 1] > -1.0)
            & (pix_coords[..., 1] < 1.0)
        )  # [n_imgs, n_rays_valid, patch_h*patch_w]

        rgb = torch.nn.functional.grid_sample(
            images.permute(0, 3, 1, 2).to(sdf.device),
            pix_coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        rgb = rgb.permute(0, 2, 3, 1)  # [n_imgs, n_rays_valid, patch_h*patch_w, 3]

        """
        # save as visualization
        import cv2
        vis_img_num = 10
        vis_patch_num = 20
        image = (
            rgb[:vis_img_num, : vis_patch_num * vis_patch_num, :, :]
            .reshape(vis_img_num, vis_patch_num, vis_patch_num, self.patch_size, self.patch_size, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, vis_patch_num * self.patch_size, 3)
        )

        def save_patch(img_id, patch_id):
            patch_rgb = rgb[img_id, patch_id].reshape(self.patch_size, self.patch_size, 3)
            cv2.imwrite("patch.png", (patch_rgb.detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1])

        cv2.imwrite("vis.png", (image.detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
        """

        return rgb, valid
