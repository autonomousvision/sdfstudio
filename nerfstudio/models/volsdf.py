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
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
from torchtyping import TensorType


from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import ErrorBoundedSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig


@dataclass
class VolSDFModelConfig(SurfaceModelConfig):
    """VolSDF Model Config"""

    _target: Type = field(default_factory=lambda: VolSDFModel)
    num_samples: int = 64
    """Number of samples after error bounded sampling"""
    num_samples_eval: int = 128
    """Number of samples per iteration used in error bounded sampling"""
    num_samples_extra: int = 32
    """Number of uniformly sampled points for training"""


class VolSDFModel(SurfaceModel):
    """VolSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: VolSDFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = ErrorBoundedSampler(
            num_samples=self.config.num_samples,
            num_samples_eval=self.config.num_samples_eval,
            num_samples_extra=self.config.num_samples_extra,
        )

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, eik_points = self.sampler(
            ray_bundle, density_fn=self.field.laplace_density, sdf_fn=self.field.get_sdf
        )
        field_outputs = self.field(ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth, "normal": normal}

        if self.training:
            grad_points = self.field.gradient(eik_points)
            outputs.update({"eik_grad": grad_points})

        return outputs

    def get_outputs_flexible(self, ray_bundle: RayBundle, additional_inputs: Dict[str, TensorType]):
        """run the model with additional inputs such as warping or rendering from unseen rays
        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            additional_inputs: addtional inputs such as images, src_idx, src_cameras

        Returns:
            dict: information needed for compute gradients
        """
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        ray_samples, eik_points = self.sampler(
            ray_bundle, density_fn=self.field.laplace_density, sdf_fn=self.field.get_sdf
        )
        field_outputs = self.field(ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        # TODO: warping and other stuff that uses additional inputs

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth, "normal": normal}

        if self.config.patch_warp_loss_mult > 0:
            # TODO visualize warped results
            # patch warping
            warped_patches, valid_mask = self.patch_warping(
                ray_samples,
                field_outputs[FieldHeadNames.SDF],
                field_outputs[FieldHeadNames.NORMAL],
                additional_inputs["src_cameras"],
                additional_inputs["src_imgs"],
                pix_indices=additional_inputs["uv"],
            )

            outputs.update({"patches": warped_patches, "patches_valid_mask": valid_mask})

        if self.training:
            grad_points = self.field.gradient(eik_points)
            outputs.update({"eik_grad": grad_points})

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            # training statics
            metrics_dict["beta"] = self.field.laplace_density.get_beta().item()
            metrics_dict["alpha"] = 1.0 / self.field.laplace_density.get_beta().item()

        return metrics_dict
