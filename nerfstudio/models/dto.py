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
import torch.nn.functional as F
from torch.nn import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.model_components.ray_samplers import (
    ErrorBoundedSampler,
    LinearDisparitySampler,
    NeuSSampler,
    PDFSampler,
)
from nerfstudio.model_components.renderers import DepthRenderer, SemanticRenderer
from nerfstudio.model_components.scene_colliders import SphereCollider
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color


@dataclass
class DtoOModelConfig(NerfactoModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: DtoOModel)
    smooth_loss_multi: float = 0.005
    """smoothness loss on surface points in unisurf"""
    sdf_field: SDFFieldConfig = SDFFieldConfig()
    """Config for SDF Field"""


class DtoOModel(NerfactoModel):
    """VolSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: DtoOModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf"))

        # Occupancy
        self.occupancy_field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        self.pdf_sampler = PDFSampler(
            include_original=True,
            single_jitter=self.config.use_single_jitter,
            histogram_padding=1e-5,
        )
        self.surface_sampler = PDFSampler(include_original=False, single_jitter=False, histogram_padding=1e-5)

        self.neus_sampler = NeuSSampler(
            num_samples=64, num_samples_importance=64, num_samples_outside=0, num_upsample_steps=4, base_variance=128
        )

        self.renderer_normal = SemanticRenderer()
        self.renderer_depth = DepthRenderer("expected")

        # for merge samples
        # self.error_bounded_sampler = ErrorBoundedSampler()
        self.error_bounded_sampler = ErrorBoundedSampler(
            num_samples=64,
            num_samples_eval=128,
            num_samples_extra=32,
        )

        # sphere collider
        self.sphere_collider = SphereCollider(radius=1.0)

        # background model
        self.bg_sampler = LinearDisparitySampler(num_samples=8)
        self.step_counter = 0
        self.anneal_end = 20000

        self.use_nerfacto = False

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        if self.use_nerfacto:
            callbacks = super().get_training_callbacks(training_callback_attributes)
        else:
            callbacks = []

        # anneal for cos in NeuS
        if self.anneal_end > 0:

            def set_anneal(step):
                anneal = min([1.0, step / self.anneal_end])
                self.occupancy_field.set_cos_anneal_ratio(anneal)

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
        param_groups["occupancy_field"] = list(self.occupancy_field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # compute near and far from from sphere with radius 1.0
        ray_bundle = self.sphere_collider(ray_bundle)

        if self.use_nerfacto:
            outputs = super().get_outputs(ray_bundle)

            # weights and ray samples from nerfacto
            base_weights = outputs["weights_list"][-1].detach()
            base_ray_samples = outputs["ray_samples_list"][-1]

            # TODO maybe interative sampling to sample more points on the surface?
            # importance samples and merge
            # occupancy_samples = self.pdf_sampler(ray_bundle, base_ray_samples, base_weights, num_samples=64)

            occupancy_samples = self.neus_sampler(
                ray_bundle, sdf_fn=self.occupancy_field.get_sdf, ray_samples=base_ray_samples
            )
        else:
            outputs = {}
            # NeuS
            # occupancy_samples = self.neus_sampler(ray_bundle, sdf_fn=self.occupancy_field.get_sdf)

            # VolSDF
            occupancy_samples, _ = self.error_bounded_sampler(
                ray_bundle, density_fn=self.occupancy_field.laplace_density, sdf_fn=self.occupancy_field.get_sdf
            )

        # occupancy unisurf
        # field_outputs = self.occupancy_field(occupancy_samples, return_occupancy=True)
        # weights, transmittance = occupancy_samples.get_weights_and_transmittance_from_alphas(
        #    field_outputs[FieldHeadNames.OCCUPANCY]
        # )

        # NeuS
        # field_outputs = self.occupancy_field(occupancy_samples, return_alphas=True)
        # weights, transmittance = occupancy_samples.get_weights_and_transmittance_from_alphas(
        #    field_outputs[FieldHeadNames.ALPHA]
        # )

        field_outputs = self.occupancy_field(occupancy_samples)
        weights, transmittance = occupancy_samples.get_weights_and_transmitance(field_outputs[FieldHeadNames.DENSITY])

        if self.training:
            # we should sample here before we change the near-far plane for ray_bundle
            # sample surface points according to distribution
            surface_samples = self.surface_sampler(ray_bundle, occupancy_samples, weights, num_samples=8)

            self.step_counter += 1
            if self.step_counter % 5000 == 0:
                from nerfstudio.model_components.ray_samplers import save_points
                from nerfstudio.utils.marching_cubes import get_surface_occupancy

                save_points("a.ply", surface_samples.frustums.get_positions().reshape(-1, 3).detach().cpu().numpy())
                get_surface_occupancy(occupancy_fn=lambda x: self.occupancy_field.forward_geonetwork(x)[:, 0])
                # breakpoint()

        bg_transmittance = transmittance[:, -1, :]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=occupancy_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # sample inversely from far to 100 and points and forwart the bg model
        ray_bundle.nears = ray_bundle.fars
        ray_bundle.fars = torch.ones_like(ray_bundle.fars) * 1000.0

        bg_ray_samples = self.bg_sampler(ray_bundle)

        # use the same background model for both density field and occupancy field
        field_outputs_bg = self.field(bg_ray_samples)
        bg_weights = bg_ray_samples.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

        bg_rgb = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=bg_weights)
        bg_depth = self.renderer_depth(weights=bg_weights, ray_samples=bg_ray_samples)
        bg_accumulation = self.renderer_accumulation(weights=bg_weights)

        bg_outputs = {
            "bg_rgb": bg_rgb,
            "bg_accumulation": bg_accumulation,
            "bg_depth": bg_depth,
            "bg_weights": bg_weights,
        }

        # merge background color to forgound color
        rgb = rgb + bg_transmittance * bg_rgb
        outputs_occupancy = {
            "orgb": rgb,
            "oaccumulation": accumulation,
            "odepth": depth,
            "onormal": normal,
            "oweights": weights,
        }

        if self.use_nerfacto:
            # merge background color to forgound color of density field
            outputs["rgb"] = outputs["rgb"] + outputs["transmittance"][:, -1, :] * bg_rgb

        if self.training:
            surface_points = surface_samples.frustums.get_positions().reshape(-1, 3)

            surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.001
            pp = torch.cat([surface_points, surface_points_neig], dim=0)
            surface_grad = self.occupancy_field.gradient(pp)
            surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)

            outputs_occupancy.update({"surface_points_normal": surface_points_normal, "surface_grad": surface_grad})

        outputs.update(outputs_occupancy)
        outputs.update(bg_outputs)

        return outputs

    def get_metrics_dict(self, outputs, batch):
        if self.use_nerfacto:
            metrics_dict = super().get_metrics_dict(outputs, batch)
        else:
            metrics_dict = {}

        image = batch["image"].to(self.device)
        metrics_dict["opsnr"] = self.psnr(outputs["orgb"], image)

        if self.training:
            # training statics
            metrics_dict["s_val"] = self.occupancy_field.deviation_network.get_variance().item()
            metrics_dict["inv_s"] = 1.0 / self.occupancy_field.deviation_network.get_variance().item()

            # training statics for volsdf
            metrics_dict["beta"] = self.occupancy_field.laplace_density.get_beta().item()
            metrics_dict["alpha"] = 1.0 / self.occupancy_field.laplace_density.get_beta().item()

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if self.use_nerfacto:
            loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        else:
            loss_dict = {}

        image = batch["image"].to(self.device)
        loss_dict["orgb_loss"] = self.rgb_loss(image, outputs["orgb"])

        if self.training and "sky" in batch:
            sky_label = 1.0 - batch["sky"].float().to(self.device)
            if self.use_nerfacto:
                density_field_weights = outputs["weights_list"][-1]
                loss_dict["sky_loss"] = (
                    F.binary_cross_entropy(density_field_weights.sum(dim=1).clip(1e-3, 1.0 - 1e-3), sky_label) * 0.01
                )

            occupancy_field_weights = outputs["oweights"]
            loss_dict["osky_loss"] = (
                F.binary_cross_entropy(occupancy_field_weights.sum(dim=1).clip(1e-3, 1.0 - 1e-3), sky_label) * 0.01
            )

        if self.training:
            surface_points_normal = outputs["surface_points_normal"]
            N = surface_points_normal.shape[0] // 2

            diff_norm = torch.norm(surface_points_normal[:N] - surface_points_normal[N:], dim=-1)
            loss_dict["normal_smoothness_loss"] = torch.mean(diff_norm) * 0.0001

            # eikonal loss
            surface_points_grad = outputs["surface_grad"]
            loss_dict["eikonal_loss"] = ((surface_points_grad.norm(2, dim=-1) - 1) ** 2).mean() * 0.0001

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        if self.use_nerfacto:
            metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        else:
            metrics_dict, images_dict = {}, {}

        image = batch["image"].to(self.device)
        rgb = outputs["orgb"]
        acc = colormaps.apply_colormap(outputs["oaccumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["odepth"],
            accumulation=outputs["oaccumulation"],
        )

        normal = outputs["onormal"]
        # don't need to normalize here
        # normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
        normal = (normal + 1.0) / 2.0
        combined_normal = torch.cat([normal], dim=1)

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict.update({"opsnr": float(psnr.item()), "ossim": float(ssim)})  # type: ignore
        metrics_dict["olpips"] = float(lpips)

        images_dict.update(
            {"oimg": combined_rgb, "oaccumulation": combined_acc, "odepth": combined_depth, "onormal": combined_normal}
        )

        return metrics_dict, images_dict
