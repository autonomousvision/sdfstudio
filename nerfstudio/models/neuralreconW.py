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
from typing import List, Type

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.model_components.ray_samplers import NeuralReconWSampler
from nerfstudio.model_components.scene_colliders import SphereCollider
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig


@dataclass
class NeuralReconWModelConfig(NeuSModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: NeuralReconWModel)


class NeuralReconWModel(NeuSModel):
    """VolSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: NeuralReconWModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # voxel surface bybrid sampler from NeuralReconW
        self.sampler = NeuralReconWSampler(
            aabb=self.scene_box.aabb, coarse_binary_grid=self.scene_box.coarse_binary_gird
        )
        # Neural Reconstruction in the wild use sphere collider so we overwrite it here
        self.collider = SphereCollider(radius=1.0, soft_intersection=False)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # add sampler call backs
        sdf_fn = lambda x: self.field.forward_geonetwork(x)[:, 0].contiguous()
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_binary_grid,
                kwargs={"sdf_fn": sdf_fn},
            )
        )

        return callbacks
