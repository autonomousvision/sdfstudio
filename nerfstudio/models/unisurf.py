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

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import UniSurfSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig


@dataclass
class UniSurfModelConfig(SurfaceModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: UniSurfModel)
    eikonal_loss_mult: float = 0.0
    """overwirte eikonal loss because it's not need for unisurf"""
    smooth_loss_multi: float = 0.005
    """smoothness loss on surface points in unisurf"""
    num_samples_interval: int = 64
    """Number of uniform samples"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for backgound"""
    num_samples_importance: int = 32
    """Number of important samples"""
    num_marching_steps: int = 256
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    perturb: bool = True
    """use to use perturb for the sampled points"""


class UniSurfModel(SurfaceModel):
    """VolSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: UniSurfModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = UniSurfSampler(
            num_samples_interval=self.config.num_samples_interval,
            num_samples_outside=self.config.num_samples_outside,
            num_samples_importance=self.config.num_samples_importance,
            num_marching_steps=self.config.num_marching_steps,
        )

        """
        self.sampler = NeuSSampler(
            num_samples=self.config.num_samples,
            num_samples_importance=self.config.num_samples_importance,
            num_samples_outside=self.config.num_samples_outside,
            num_upsample_steps=self.config.num_up_sample_steps,
            base_variance=self.config.base_variance,
        )
        """

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.step_cb,
            )
        )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, surface_points = self.sampler(
            ray_bundle, occupancy_fn=self.field.get_occupancy, sdf_fn=self.field.get_sdf, return_surface_points=True
        )
        field_outputs = self.field(ray_samples, return_occupancy=True)
        weights = ray_samples.get_weights_from_alphas(field_outputs[FieldHeadNames.OCCUPANCY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth, "normal": normal}

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]

            surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01
            pp = torch.cat([surface_points, surface_points_neig], dim=0)
            surface_grad = self.field.gradient(pp)
            surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)

            outputs.update({"eik_grad": grad_points, "surface_points_normal": surface_points_normal})

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
        ray_samples, surface_points = self.sampler(
            ray_bundle, occupancy_fn=self.field.get_occupancy, sdf_fn=self.field.get_sdf, return_surface_points=True
        )
        field_outputs = self.field(ray_samples, return_occupancy=True)
        weights = ray_samples.get_weights_from_alphas(field_outputs[FieldHeadNames.OCCUPANCY])
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
            grad_points = field_outputs[FieldHeadNames.GRADIENT]

            surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01
            pp = torch.cat([surface_points, surface_points_neig], dim=0)
            surface_grad = self.field.gradient(pp)
            surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)

            outputs.update({"eik_grad": grad_points, "surface_points_normal": surface_points_normal})

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            # training statics
            metrics_dict["delta"] = self.sampler.delta

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # TODO move to base model?
        if self.training and self.config.smooth_loss_multi > 0.0:
            surface_points_normal = outputs["surface_points_normal"]
            N = surface_points_normal.shape[0] // 2

            diff_norm = torch.norm(surface_points_normal[:N] - surface_points_normal[N:], dim=-1)
            loss_dict["normal_smoothness_loss"] = torch.mean(diff_norm) * self.config.smooth_loss_multi

        return loss_dict
