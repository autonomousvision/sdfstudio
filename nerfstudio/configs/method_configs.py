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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import (
    Config,
    SchedulerConfig,
    TrainerConfig,
    ViewerConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import (
    FlexibleDataManagerConfig,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.uniscene_dataparser import UniSceneDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialSchedulerConfig,
    MultiStepSchedulerConfig,
    NeuSSchedulerConfig,
)
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.models.dto import DtoOModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neuralreconW import NeuralReconWModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.neus_acc import NeuSAccModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.unisurf import UniSurfModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.models.volsdf import VolSDFModelConfig
from nerfstudio.pipelines.base_pipeline import (
    FlexibleInputPipelineConfig,
    VanillaPipelineConfig,
)
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig

method_configs: Dict[str, Config] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for bounded synthetic data.",
    "mipnerf": "High quality model for bounded scenes. (slow)",
    "semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "vanilla-nerf": "Original NeRF model. (slow)",
    "volsdf": "Implementation of VolSDF.",
    "monosdf": "Implementation of MonoSDF.",
    "geo-volsdf": "Implementation of patch warping from GeoNeuS with VolSDF.",
    "neus": "Implementation of NeuS.",
    "mono-neus": "Implementation of MonoSDF with NeuS rendering formulation.",
    "geo-neus": "Implementation of patch warping from GeoNeuS with NeuS.",
    "unisurf": "Implementation of UniSurf.",
    "mono-unisurf": "Implementation of MonoSDF with unisurf rendering formulation.",
    "geo-unisurf": "Implementation of patch warping from GeoNeuS with UniSurf.",
    "tensorf": "tensorf",
    "dnerf": "Dynamic-NeRF model. (slow)",
    "dto": "Occupancy field with density guided sampling",
    "neusW": "Implementation of Neural Reconstruction in the wild",
    "neus-acc": "Implementation of NeuS with empty space skipping.",
}

method_configs["geo-volsdf"] = Config(
    method_name="geo-volsdf",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=200001,
        mixed_precision=False,
    ),
    pipeline=FlexibleInputPipelineConfig(
        datamanager=FlexibleDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=VolSDFModelConfig(patch_warp_loss_mult=0.1, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(
                max_steps=1000000
            ),  # set max_steps to a large value so it never step and we will use the last_lr form the pretrained model
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["monosdf"] = Config(
    method_name="monosdf",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=200000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=VolSDFModelConfig(mono_depth_loss_mult=0.1, mono_normal_loss_mult=0.05, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=200000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["volsdf"] = Config(
    method_name="volsdf",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=VolSDFModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=100000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=100000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["geo-neus"] = Config(
    method_name="geo-neus",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=200000,
        mixed_precision=False,
    ),
    pipeline=FlexibleInputPipelineConfig(
        datamanager=FlexibleDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(patch_warp_loss_mult=0.1, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["mono-neus"] = Config(
    method_name="mono-neus",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(mono_depth_loss_mult=0.1, mono_normal_loss_mult=0.05, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neus"] = Config(
    method_name="neus",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["unisurf"] = Config(
    method_name="unisurf",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=UniSurfModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["mono-unisurf"] = Config(
    method_name="mono-unisurf",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=UniSurfModelConfig(mono_depth_loss_mult=0.1, mono_normal_loss_mult=0.05, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["geo-unisurf"] = Config(
    method_name="geo-unisurf",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=FlexibleInputPipelineConfig(
        datamanager=FlexibleDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=UniSurfModelConfig(patch_warp_loss_mult=0.1, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["dto"] = Config(
    method_name="dto",
    trainer=TrainerConfig(
        steps_per_eval_image=2000,
        steps_per_eval_batch=5000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=DtoOModelConfig(eval_num_rays_per_chunk=1 << 10),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
        "occupancy_field": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neusW"] = Config(
    method_name="neusW",
    trainer=TrainerConfig(
        steps_per_eval_image=2000,
        steps_per_eval_batch=5000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuralReconWModelConfig(eval_num_rays_per_chunk=1 << 10),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neus-acc"] = Config(
    method_name="neus-acc",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=UniSceneDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSAccModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


method_configs["nerfacto"] = Config(
    method_name="nerfacto",
    trainer=TrainerConfig(
        steps_per_eval_batch=5000, steps_per_save=20000, max_num_iterations=300000, mixed_precision=True
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["instant-ngp"] = Config(
    method_name="instant-ngp",
    trainer=TrainerConfig(
        steps_per_eval_batch=5000,
        steps_per_eval_image=5000,
        steps_per_save=20000,
        max_num_iterations=20001,
        mixed_precision=True,
        steps_per_eval_all_images=20000,
    ),
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(render_step_size=0.005, eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=20000),
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)

method_configs["mipnerf"] = Config(
    method_name="mipnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=BlenderDataParserConfig(), train_num_rays_per_batch=8192),
        model=VanillaModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=8192,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

method_configs["semantic-nerfw"] = Config(
    method_name="semantic-nerfw",
    trainer=TrainerConfig(steps_per_eval_batch=500, steps_per_save=2000, mixed_precision=True),
    pipeline=VanillaPipelineConfig(
        datamanager=SemanticDataManagerConfig(
            dataparser=FriendsDataParserConfig(), train_num_rays_per_batch=4096, eval_num_rays_per_batch=8192
        ),
        model=SemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="viewer",
)

method_configs["vanilla-nerf"] = Config(
    method_name="vanilla-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=VanillaModelConfig(_target=NeRFModel),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["tensorf"] = Config(
    method_name="tensorf",
    trainer=TrainerConfig(mixed_precision=False),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=TensoRFModelConfig(),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.001),
            "scheduler": SchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "encodings": {
            "optimizer": AdamOptimizerConfig(lr=0.02),
            "scheduler": SchedulerConfig(lr_final=0.002, max_steps=30000),
        },
    },
)

method_configs["dnerf"] = Config(
    method_name="dnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=DNeRFDataParserConfig()),
        model=VanillaModelConfig(
            _target=NeRFModel,
            enable_temporal_distortion=True,
            temporal_distortion_params={"kind": TemporalDistortionKind.DNERF},
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
