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
from typing import Dict, List, Type

import torch

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

        # can't use eikonal loss in Unisurf? or we could use learnable paramter to transform sdf to occupancy
        assert self.config.eikonal_loss_mult == 0.0

        self.sampler = UniSurfSampler(
            num_samples_interval=self.config.num_samples_interval,
            num_samples_outside=self.config.num_samples_outside,
            num_samples_importance=self.config.num_samples_importance,
            num_marching_steps=self.config.num_marching_steps,
        )

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

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        ray_samples, surface_points = self.sampler(
            ray_bundle, occupancy_fn=self.field.get_occupancy, sdf_fn=self.field.get_sdf, return_surface_points=True
        )
        field_outputs = self.field(ray_samples, return_occupancy=True)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.OCCUPANCY]
        )
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "surface_points": surface_points,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }
        return samples_and_field_outputs

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            metrics_dict["delta"] = self.sampler.delta

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # TODO move to base model as other model could also use it?
        if self.training and self.config.smooth_loss_multi > 0.0:
            surface_points = outputs["surface_points"]

            surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01
            pp = torch.cat([surface_points, surface_points_neig], dim=0)
            surface_grad = self.field.gradient(pp)
            surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)

            N = surface_points_normal.shape[0] // 2

            diff_norm = torch.norm(surface_points_normal[:N] - surface_points_normal[N:], dim=-1)
            loss_dict["normal_smoothness_loss"] = torch.mean(diff_norm) * self.config.smooth_loss_multi

        return loss_dict
