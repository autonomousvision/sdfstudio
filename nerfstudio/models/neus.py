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
Implementation of MonoSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.monosdf import MonoSDFModel, MonoSDFModelConfig
from nerfstudio.model_components.ray_samplers import NeuSSampler


@dataclass
class NeuSModelConfig(MonoSDFModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NeuSModel)
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for backgound"""
    num_up_sample_steps = 4
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    perturb: bool = True
    """use to use perturb for the sampled points"""


class NeuSModel(MonoSDFModel):
    """MonoSDF model

    Args:
        config: MonoSDF configuration to instantiate model
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
        )

        self.anneal_end = 50000

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

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        field_outputs = self.field(ray_samples, return_alphas=True)
        weights = ray_samples.get_weights_from_alphas(field_outputs[FieldHeadNames.ALPHA])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth, "normal": normal}

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
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
