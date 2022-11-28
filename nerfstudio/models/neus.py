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
Implementation of NeuS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler, NeuSSampler
from nerfstudio.model_components.scene_colliders import SphereCollider
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig


@dataclass
class NeuSModelConfig(SurfaceModelConfig):
    """NeuS Model Config"""

    _target: Type = field(default_factory=lambda: NeuSModel)
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for backgound"""
    num_up_sample_steps: int = 4
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    base_variance: float = 64
    """fixed base variance in NeuS sampler, the inv_s will be base * 2 ** iter during upsample"""
    perturb: bool = True
    """use to use perturb for the sampled points"""


class NeuSModel(SurfaceModel):
    """NeuS model

    Args:
        config: NeuS configuration to instantiate model
    """

    config: NeuSModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = NeuSSampler(
            num_samples=self.config.num_samples,
            num_samples_importance=self.config.num_samples_importance,
            num_samples_outside=self.config.num_samples_outside,
            num_upsample_steps=self.config.num_up_sample_steps,
            base_variance=self.config.base_variance,
        )

        # TODO make background model optional
        # background model
        self.field_background = TCNNNerfactoField(
            self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )
        self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

        self.anneal_end = 50000

        # TODO change to also suppors other collider such as near and far or bbox colider
        # sphere collider
        self.sphere_collider = SphereCollider(radius=1.0)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        # anneal for cos in NeuS
        if self.anneal_end > 0:

            def set_anneal(step):
                anneal = min([1.0, step / self.anneal_end])
                self.field.set_cos_anneal_ratio(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )

        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["field_background"] = list(self.field_background.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # compute near and far from from sphere with radius 1.0
        ray_bundle = self.sphere_collider(ray_bundle)

        ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        field_outputs = self.field(ray_samples, return_alphas=True)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
        bg_transmittance = transmittance[:, -1, :]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # TODO remove hard-coded far value
        # sample inversely from far to 1000 and points and forward the bg model
        ray_bundle.nears = ray_bundle.fars
        ray_bundle.fars = torch.ones_like(ray_bundle.fars) * 1000.0

        ray_samples_bg = self.sampler_bg(ray_bundle)
        # use the same background model for both density field and occupancy field
        field_outputs_bg = self.field_background(ray_samples_bg)
        weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

        rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
        depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
        accumulation_bg = self.renderer_accumulation(weights=weights_bg)

        # merge background color to forgound color
        rgb = rgb + bg_transmittance * rgb_bg

        bg_outputs = {
            "bg_rgb": rgb_bg,
            "bg_accumulation": accumulation_bg,
            "bg_depth": depth_bg,
            "bg_weights": weights_bg,
        }

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth, "normal": normal, "weights": weights}

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})

        outputs.update(bg_outputs)
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
        ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        field_outputs = self.field(ray_samples, return_alphas=True)
        weights = ray_samples.get_weights_from_alphas(field_outputs[FieldHeadNames.ALPHA])
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
            outputs.update({"eik_grad": grad_points})

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            # training statics
            metrics_dict["s_val"] = self.field.deviation_network.get_variance().item()
            metrics_dict["inv_s"] = 1.0 / self.field.deviation_network.get_variance().item()

        return metrics_dict
