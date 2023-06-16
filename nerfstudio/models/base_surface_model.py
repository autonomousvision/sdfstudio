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
Implementation of Base surface model.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchtyping import TensorType
from typing_extensions import Literal
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    MultiViewLoss,
    ScaleAndShiftInvariantLoss,
    SensorDepthLoss,
    compute_scale_and_shift,
    monosdf_normal_loss,
)
from nerfstudio.model_components.patch_warping import PatchWarping
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import (
    AABBBoxCollider,
    NearFarCollider,
    SphereCollider,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color


@dataclass
class SurfaceModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SurfaceModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 4.0
    """How far along the ray to stop sampling."""
    far_plane_bg: float = 1000.0
    """How far along the ray to stop sampling of the background model."""
    background_color: Literal["random", "last_sample", "white", "black"] = "black"
    """Whether to randomize the background color."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    eikonal_loss_mult: float = 0.1
    """Eikonal loss multiplier."""
    fg_mask_loss_mult: float = 0.01
    """Foreground mask loss multiplier."""
    mono_normal_loss_mult: float = 0.0
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    patch_warp_loss_mult: float = 0.0
    """Multi-view consistency warping loss multiplier."""
    patch_size: int = 11
    """Multi-view consistency warping loss patch size."""
    patch_warp_angle_thres: float = 0.3
    """Threshold for valid homograph of multi-view consistency warping loss"""
    min_patch_variance: float = 0.01
    """Threshold for minimal patch variance"""
    topk: int = 4
    """Number of minimal patch consistency selected for training"""
    sensor_depth_truncation: float = 0.015
    """Sensor depth trunction, default value is 0.015 which means 5cm with a rough scale value 0.3 (0.015 = 0.05 * 0.3)"""
    sensor_depth_l1_loss_mult: float = 0.0
    """Sensor depth L1 loss multiplier."""
    sensor_depth_freespace_loss_mult: float = 0.0
    """Sensor depth free space loss multiplier."""
    sensor_depth_sdf_loss_mult: float = 0.0
    """Sensor depth sdf loss multiplier."""
    sparse_points_sdf_loss_mult: float = 0.0
    """sparse point sdf loss multiplier"""
    sdf_field: SDFFieldConfig = SDFFieldConfig()
    """Config for SDF Field"""
    background_model: Literal["grid", "mlp", "none"] = "mlp"
    """background models"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for backgound"""
    periodic_tvl_mult: float = 0.0
    """Total variational loss mutliplier"""
    overwrite_near_far_plane: bool = False
    """whether to use near and far collider from command line"""
    scene_contraction_norm: Literal["inf", "l2"] = "inf"
    """Which norm to use for the scene contraction."""


class SurfaceModel(Model):
    """Base surface model

    Args:
        config: Base surface model configuration to instantiate model
    """

    config: SurfaceModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.scene_contraction_norm == "inf":
            order = float("inf")
        elif self.config.scene_contraction_norm == "l2":
            order = None
        else:
            raise ValueError("Invalid scene contraction norm")

        self.scene_contraction = SceneContraction(order=order)

        # Can we also use contraction for sdf?
        # Fields
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        # Collider
        if self.scene_box.collider_type == "near_far":
            self.collider = NearFarCollider(near_plane=self.scene_box.near, far_plane=self.scene_box.far)
        elif self.scene_box.collider_type == "box":
            self.collider = AABBBoxCollider(self.scene_box, near_plane=self.scene_box.near)
        elif self.scene_box.collider_type == "sphere":
            # TODO do we also use near if the ray don't intersect with the sphere
            self.collider = SphereCollider(radius=self.scene_box.radius, soft_intersection=True)
        else:
            raise NotImplementedError

        # command line near and far has highest priority
        if self.config.overwrite_near_far_plane:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # background model
        if self.config.background_model == "grid":
            self.field_background = TCNNNerfactoField(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            )
        elif self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )

            self.field_background = NeRFField(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
            )
        else:
            # dummy background model
            self.field_background = Parameter(torch.ones(1), requires_grad=False)

        self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normal = SemanticRenderer()
        # patch warping
        self.patch_warping = PatchWarping(
            patch_size=self.config.patch_size, valid_angle_thres=self.config.patch_warp_angle_thres
        )

        # losses
        self.rgb_loss = L1Loss()
        self.eikonal_loss = MSELoss()
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        self.patch_loss = MultiViewLoss(
            patch_size=self.config.patch_size, topk=self.config.topk, min_patch_variance=self.config.min_patch_variance
        )
        self.sensor_depth_loss = SensorDepthLoss(truncation=self.config.sensor_depth_truncation)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        else:
            param_groups["field_background"] = list(self.field_background)
        return param_groups

    @abstractmethod
    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        """_summary_

        Args:
            ray_bundle (RayBundle): _description_
            return_samples (bool, optional): _description_. Defaults to False.
        """

    def get_foreground_mask(self, ray_samples: RaySamples) -> TensorType:
        """_summary_

        Args:
            ray_samples (RaySamples): _description_
        """
        # TODO support multiple foreground type: box and sphere
        inside_sphere_mask = (ray_samples.frustums.get_start_positions().norm(dim=-1, keepdim=True) < 1.0).float()
        return inside_sphere_mask

    def forward_background_field_and_merge(self, ray_samples: RaySamples, field_outputs: Dict) -> Dict:
        """_summary_

        Args:
            ray_samples (RaySamples): _description_
            field_outputs (Dict): _description_
        """

        inside_sphere_mask = self.get_foreground_mask(ray_samples)
        # TODO only forward the points that are outside the sphere if there is a background model

        field_outputs_bg = self.field_background(ray_samples)
        field_outputs_bg[FieldHeadNames.ALPHA] = ray_samples.get_alphas(field_outputs_bg[FieldHeadNames.DENSITY])

        field_outputs[FieldHeadNames.ALPHA] = (
            field_outputs[FieldHeadNames.ALPHA] * inside_sphere_mask
            + (1.0 - inside_sphere_mask) * field_outputs_bg[FieldHeadNames.ALPHA]
        )
        field_outputs[FieldHeadNames.RGB] = (
            field_outputs[FieldHeadNames.RGB] * inside_sphere_mask
            + (1.0 - inside_sphere_mask) * field_outputs_bg[FieldHeadNames.RGB]
        )

        # TODO make everything outside the sphere to be 0
        return field_outputs

    def get_outputs(self, ray_bundle: RayBundle) -> Dict:
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        # remove the rays that don't intersect with the surface
        # hit = (field_outputs[FieldHeadNames.SDF] > 0.0).any(dim=1) & (field_outputs[FieldHeadNames.SDF] < 0).any(dim=1)
        # depth[~hit] = 10000.0

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            "ray_points": self.scene_contraction(
                ray_samples.frustums.get_start_positions()
            ),  # used for creating visiblity mask
            "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
        }

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            points_norm = field_outputs["points_norm"]
            outputs.update({"eik_grad": grad_points, "points_norm": points_norm})

            # TODO volsdf use different point set for eikonal loss
            # grad_points = self.field.gradient(eik_points)
            # outputs.update({"eik_grad": grad_points})

            outputs.update(samples_and_field_outputs)

        # TODO how can we move it to neus_facto without out of memory
        if "weights_list" in samples_and_field_outputs:
            weights_list = samples_and_field_outputs["weights_list"]
            ray_samples_list = samples_and_field_outputs["ray_samples_list"]

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs

    def get_outputs_flexible(self, ray_bundle: RayBundle, additional_inputs: Dict[str, TensorType]) -> Dict:
        """run the model with additional inputs such as warping or rendering from unseen rays
        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            additional_inputs: addtional inputs such as images, src_idx, src_cameras

        Returns:
            dict: information needed for compute gradients
        """
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        outputs = self.get_outputs(ray_bundle)

        ray_samples = outputs["ray_samples"]
        field_outputs = outputs["field_outputs"]

        if self.config.patch_warp_loss_mult > 0:
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

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

            # foreground mask loss
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )

            # monocular normal loss
            if "normal" in batch and self.config.mono_normal_loss_mult > 0.0:
                normal_gt = batch["normal"].to(self.device)
                normal_pred = outputs["normal"]
                loss_dict["normal_loss"] = (
                    monosdf_normal_loss(normal_pred, normal_gt) * self.config.mono_normal_loss_mult
                )

            # monocular depth loss
            if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
                # TODO check it's true that's we sample from only a single image
                # TODO only supervised pixel that hit the surface and remove hard-coded scaling for depth
                depth_gt = batch["depth"].to(self.device)[..., None]
                depth_pred = outputs["depth"]

                mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
                loss_dict["depth_loss"] = (
                    self.depth_loss(depth_pred.reshape(1, 32, -1), (depth_gt * 50 + 0.5).reshape(1, 32, -1), mask)
                    * self.config.mono_depth_loss_mult
                )

            # sensor depth loss
            if "sensor_depth" in batch and (
                self.config.sensor_depth_l1_loss_mult > 0.0
                or self.config.sensor_depth_freespace_loss_mult > 0.0
                or self.config.sensor_depth_sdf_loss_mult > 0.0
            ):
                l1_loss, free_space_loss, sdf_loss = self.sensor_depth_loss(batch, outputs)

                loss_dict["sensor_l1_loss"] = l1_loss * self.config.sensor_depth_l1_loss_mult
                loss_dict["sensor_freespace_loss"] = free_space_loss * self.config.sensor_depth_freespace_loss_mult
                loss_dict["sensor_sdf_loss"] = sdf_loss * self.config.sensor_depth_sdf_loss_mult

            # multi-view photoconsistency loss as Geo-NeuS
            if "patches" in outputs and self.config.patch_warp_loss_mult > 0.0:
                patches = outputs["patches"]
                patches_valid_mask = outputs["patches_valid_mask"]

                loss_dict["patch_loss"] = (
                    self.patch_loss(patches, patches_valid_mask) * self.config.patch_warp_loss_mult
                )

            # sparse points sdf loss
            if "sparse_sfm_points" in batch and self.config.sparse_points_sdf_loss_mult > 0.0:
                sparse_sfm_points = batch["sparse_sfm_points"].to(self.device)
                sparse_sfm_points_sdf = self.field.forward_geonetwork(sparse_sfm_points)[:, 0].contiguous()
                loss_dict["sparse_sfm_points_sdf_loss"] = (
                    torch.mean(torch.abs(sparse_sfm_points_sdf)) * self.config.sparse_points_sdf_loss_mult
                )

            # total variational loss for multi-resolution periodic feature volume
            if self.config.periodic_tvl_mult > 0.0:
                assert self.field.config.encoding_type == "periodic"
                loss_dict["tvl_loss"] = self.field.encoding.get_total_variation_loss() * self.config.periodic_tvl_mult

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])

        normal = outputs["normal"]
        # don't need to normalize here
        # normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
        normal = (normal + 1.0) / 2.0

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        if "depth" in batch:
            depth_gt = batch["depth"].to(self.device)
            depth_pred = outputs["depth"]

            # align to predicted depth and normalize
            scale, shift = compute_scale_and_shift(
                depth_pred[None, ..., 0], depth_gt[None, ...], depth_gt[None, ...] > 0.0
            )
            depth_pred = depth_pred * scale + shift

            combined_depth = torch.cat([depth_gt[..., None], depth_pred], dim=1)
            combined_depth = colormaps.apply_depth_colormap(combined_depth)
        else:
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
            combined_depth = torch.cat([depth], dim=1)

        if "normal" in batch:
            normal_gt = (batch["normal"].to(self.device) + 1.0) / 2.0
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        else:
            combined_normal = torch.cat([normal], dim=1)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "normal": combined_normal,
        }

        if "sensor_depth" in batch:
            sensor_depth = batch["sensor_depth"]
            depth_pred = outputs["depth"]

            combined_sensor_depth = torch.cat([sensor_depth[..., None], depth_pred], dim=1)
            combined_sensor_depth = colormaps.apply_depth_colormap(combined_sensor_depth)
            images_dict["sensor_depth"] = combined_sensor_depth

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        return metrics_dict, images_dict
