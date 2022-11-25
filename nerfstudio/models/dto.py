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

import nerfacc
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
from nerfstudio.model_components.losses import L1Loss
from nerfstudio.model_components.ray_samplers import (
    ErrorBoundedSampler,
    LinearDisparitySampler,
    NeuSSampler,
    PDFSampler,
    UniformSampler,
    UniSurfSampler,
    save_points,
)
from nerfstudio.model_components.renderers import DepthRenderer, SemanticRenderer
from nerfstudio.model_components.scene_colliders import SphereCollider
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.marching_cubes import get_surface_occupancy


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

        # create occupancy grid from scene_bbox
        aabb_scale = 1.0
        aabb = [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]]
        aabb = torch.tensor(aabb, dtype=torch.float32)

        self.grid = nerfacc.OccupancyGrid(aabb.reshape(-1), resolution=32)
        self._binary = self.scene_box.reshape(32, 32, 32).contiguous()
        self._binary_fine = None
        self.rank = torch.distributed.get_rank()
        print("self", self.rank)

        # Occupancy
        self.occupancy_field = self.config.sdf_field.setup(
            aabb=aabb,
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

        self.uniform_sampler = UniformSampler(single_jitter=False)

        self.neus_sampler = NeuSSampler(
            num_samples=8, num_samples_importance=16, num_samples_outside=0, num_upsample_steps=2, base_variance=512
        )

        self.renderer_normal = SemanticRenderer()
        self.renderer_depth = DepthRenderer("expected")

        # for merge samples
        self.unisurf_sampler = UniSurfSampler()

        # self.error_bounded_sampler = ErrorBoundedSampler()
        self.error_bounded_sampler = ErrorBoundedSampler(
            num_samples=64,
            num_samples_eval=128,
            num_samples_extra=32,
        )

        # sphere collider
        self.sphere_collider = SphereCollider(radius=1.0)

        # background model
        self.bg_sampler = LinearDisparitySampler(num_samples=4)
        self.step_counter = 0
        self.anneal_end = 20000

        self.use_nerfacto = False
        self.method = "neus"

        self.rgb_loss = L1Loss()

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

        # near and far from occupancy grids
        packed_info, ray_indices, t_starts, t_ends = nerfacc.cuda.ray_marching(
            ray_bundle.origins.contiguous(),
            ray_bundle.directions.contiguous(),
            ray_bundle.nears[:, 0].contiguous(),
            ray_bundle.fars[:, 0].contiguous(),
            self.grid.roi_aabb.contiguous(),
            self._binary.to(ray_bundle.origins.device),
            self.grid.contraction_type.to_cpp_version(),
            1e-2,  # large value for coarse voxels
            0.0,
        )

        tt_starts = nerfacc.unpack_data(packed_info, t_starts)
        # tt_ends = nerfacc.unpack_data(packed_info, t_ends)

        hit_grid = (tt_starts > 0).any(dim=1)[:, 0]
        if hit_grid.float().sum() > 0:
            ray_bundle.nears[hit_grid] = tt_starts[hit_grid][:, 0]
            ray_bundle.fars[hit_grid] = tt_starts[hit_grid].max(dim=1)[0]

        # sample uniformly with currently nears and far
        voxel_samples = self.uniform_sampler(ray_bundle, num_samples=10)

        nears = ray_bundle.nears.clone()
        fars = ray_bundle.fars.clone()

        if self.training and self.step_counter > 5000 and self.step_counter % 5000 == 1:
            grid_size = 32
            voxel_size = 2.0 / 32
            fine_grid_size = 16
            offset = torch.linspace(-1.0, 1.0, fine_grid_size * 2 + 1, device=self.device)[1::2]
            x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
            fine_offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3) * voxel_size * 0.5

            # coarse grid coordinates
            offset = torch.linspace(-1.0 + voxel_size / 2.0, 1.0 - voxel_size / 2.0, grid_size, device=self.device)
            x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
            coarse_offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

            # xyz
            mask = torch.zeros((grid_size**3, fine_grid_size**3), dtype=torch.bool, device=self.device)

            occupied_voxel = coarse_offset_cube[self._binary.reshape(-1)]
            fine_voxel = occupied_voxel[:, None] + fine_offset_cube[None, :]
            print(fine_voxel.shape)
            fine_voxel = fine_voxel.reshape(-1, 3)
            # save_points("fine_voxel.ply", fine_voxel.cpu().numpy())

            @torch.no_grad()
            def evaluate(points):
                z = []
                for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                    z.append(self.occupancy_field.forward_geonetwork(pnts)[:, 0].contiguous())
                z = torch.cat(z, axis=0)
                return z

            sdf = evaluate(fine_voxel)
            sdf = sdf.reshape(occupied_voxel.shape[0], fine_grid_size**3)
            sdf_mask = sdf <= 0.0
            mask[self._binary.reshape(-1)] = sdf_mask

            self._binary_fine = (
                mask.reshape(grid_size, grid_size, grid_size, fine_grid_size, fine_grid_size, fine_grid_size)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(grid_size * fine_grid_size, grid_size * fine_grid_size, grid_size * fine_grid_size)
                .contiguous()
            )

            offset = torch.linspace(-1.0, 1.0, fine_grid_size * grid_size, device=self.device)
            x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
            grid_coord = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
            if self.rank == 0:
                save_points("fine_voxel_valid.ply", grid_coord[self._binary_fine.reshape(-1)].cpu().numpy())
            # breakpoint()

        if self._binary_fine is not None:
            # near and far from occupancy grids
            packed_info, ray_indices, t_starts, t_ends = nerfacc.cuda.ray_marching(
                ray_bundle.origins.contiguous(),
                ray_bundle.directions.contiguous(),
                ray_bundle.nears[:, 0].contiguous(),
                ray_bundle.fars[:, 0].contiguous(),
                self.grid.roi_aabb.contiguous(),
                self._binary_fine,
                self.grid.contraction_type.to_cpp_version(),
                1e-3,  # small value for fine voxels
                0.0,
            )

            tt_starts = nerfacc.unpack_data(packed_info, t_starts)
            # tt_ends = nerfacc.unpack_data(packed_info, t_ends, n_samples=1024)

            # update with near and far
            hit_grid = (tt_starts > 0).any(dim=1)[:, 0]
            if hit_grid.float().sum() > 0:
                ray_bundle.nears[hit_grid] = tt_starts[hit_grid][:, 0] - 0.03
                ray_bundle.fars[hit_grid] = tt_starts[hit_grid][:, 0] + 0.03
            else:
                print("waring not intersection")
            print("sampling around surfaces")

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
            if self.method == "neus":
                occupancy_samples = self.neus_sampler(ray_bundle, sdf_fn=self.occupancy_field.get_sdf)
            elif self.method == "volsdf":
                # VolSDF
                occupancy_samples, _ = self.error_bounded_sampler(
                    ray_bundle, density_fn=self.occupancy_field.laplace_density, sdf_fn=self.occupancy_field.get_sdf
                )

        # save_points("p2.ply", occupancy_samples.frustums.get_positions().detach().cpu().numpy().reshape(-1, 3))

        # merge samples
        occupancy_samples = self.unisurf_sampler.merge_ray_samples_in_eculidean(
            ray_bundle, occupancy_samples, voxel_samples
        )

        # save_points("p1.ply", voxel_samples.frustums.get_positions().detach().cpu().numpy().reshape(-1, 3))

        # save_points("p3.ply", occupancy_samples.frustums.get_positions().detach().cpu().numpy().reshape(-1, 3))
        # breakpoint()

        # save_points("p4.ply", importance_samples.frustums.get_positions().detach().cpu().numpy().reshape(-1, 3))
        # if self._step > 0:
        #    exit(-1)

        # save_points("merged.ply", merged_samples.frustums.get_start_positions().reshape(-1, 3).detach().cpu().numpy())

        # breakpoint()

        # occupancy unisurf
        # field_outputs = self.occupancy_field(occupancy_samples, return_occupancy=True)
        # weights, transmittance = occupancy_samples.get_weights_and_transmittance_from_alphas(
        #    field_outputs[FieldHeadNames.OCCUPANCY]
        # )
        # save_points("o.ply", occupancy_samples.frustums.get_positions().reshape(-1, 3).detach().cpu().numpy())
        if self.method == "neus":
            field_outputs = self.occupancy_field(occupancy_samples, return_alphas=True)
            weights, transmittance = occupancy_samples.get_weights_and_transmittance_from_alphas(
                field_outputs[FieldHeadNames.ALPHA]
            )
        elif self.method == "volsdf":
            field_outputs = self.occupancy_field(occupancy_samples)
            weights, transmittance = occupancy_samples.get_weights_and_transmitance(
                field_outputs[FieldHeadNames.DENSITY]
            )

        # save_points("a.ply", occupancy_samples.frustums.get_positions().reshape(-1, 3).detach().cpu().numpy())

        if self.training:
            # we should sample here before we change the near-far plane for ray_bundle
            # sample surface points according to distribution
            # surface_samples = self.surface_sampler(ray_bundle, occupancy_samples, weights, num_samples=8)

            self.step_counter += 1
            if self.step_counter % 5000 == 0 and self.rank == 0:

                save_points("a.ply", occupancy_samples.frustums.get_positions().reshape(-1, 3).detach().cpu().numpy())
                get_surface_occupancy(
                    occupancy_fn=lambda x: self.occupancy_field.forward_geonetwork(x)[:, 0], device=self.device
                )
                # breakpoint()

        bg_transmittance = transmittance[:, -1, :]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=occupancy_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # save rendered points for visualization
        pts = self.renderer_normal(semantics=occupancy_samples.frustums.get_positions(), weights=weights)
        hit_mask = (field_outputs[FieldHeadNames.SDF] > 0.0).any(dim=1) & (field_outputs[FieldHeadNames.SDF] < 0.0).any(
            dim=1
        )
        pts = pts[hit_mask[:, 0]]
        # save_points("a.ply", pts.reshape(-1, 3).detach().cpu().numpy())
        # breakpoint()
        if pts.shape[0] > 0:
            surface_sdf = self.occupancy_field.forward_geonetwork(pts)[:, 0]
        else:
            surface_sdf = None

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

        if self.training:
            outputs_occupancy.update({"surface_sdf": surface_sdf})

        if self.use_nerfacto:
            # merge background color to forgound color of density field
            outputs["rgb"] = outputs["rgb"] + outputs["transmittance"][:, -1, :] * bg_rgb

        if self.training:
            # surface_points = surface_samples.frustums.get_positions().reshape(-1, 3)

            # surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.001
            # pp = torch.cat([surface_points, surface_points_neig], dim=0)
            # surface_grad = self.occupancy_field.gradient(pp)
            # surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)

            # outputs_occupancy.update({"surface_points_normal": surface_points_normal, "surface_grad": surface_grad})
            outputs_occupancy.update({"surface_grad": field_outputs[FieldHeadNames.GRADIENT]})

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
            # surface_points_normal = outputs["surface_points_normal"]
            # N = surface_points_normal.shape[0] // 2

            # diff_norm = torch.norm(surface_points_normal[:N] - surface_points_normal[N:], dim=-1)
            # loss_dict["normal_smoothness_loss"] = torch.mean(diff_norm) * 0.0001

            # eikonal loss
            surface_points_grad = outputs["surface_grad"]
            loss_dict["eikonal_loss"] = ((surface_points_grad.norm(2, dim=-1) - 1) ** 2).mean() * 0.0001

            # surface points loss
            surface_points_sdf = outputs["surface_sdf"]
            if surface_points_sdf is not None:
                loss_dict["surface_sdf_loss"] = torch.abs(surface_points_sdf).mean() * 0.0

            sparse_pts = batch["sparse_pts"].to(self.device)
            sparse_pts, pts_weights = sparse_pts[:, :3], sparse_pts[:, 3:]

            # filter norm
            in_sphere = torch.norm(sparse_pts, dim=-1) < 1.0
            sparse_pts = sparse_pts[in_sphere]
            pts_weights = pts_weights[in_sphere]
            pts_weights = torch.ones_like(pts_weights)
            # print(in_sphere.float().mean())
            # save_points("sa.ply", sparse_pts.cpu().numpy())
            # breakpoint()
            if sparse_pts.shape[0] > 0:
                sparse_pts_sdf = self.occupancy_field.forward_geonetwork(sparse_pts)[:, 0]
                loss_dict["sparse_pts_loss"] = (torch.abs(sparse_pts_sdf) * pts_weights).mean() * 0.0

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
