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
Collection of sampling strategies
"""

import math
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import nerfacc
import torch
from nerfacc import OccupancyGrid
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples


class Sampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

    @abstractmethod
    def generate_ray_samples(self) -> RaySamples:
        """Generate Ray Samples"""

    def forward(self, *args, **kwargs) -> RaySamples:
        """Generate ray samples"""
        return self.generate_ray_samples(*args, **kwargs)


class SpacedSampler(Sampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples accoring to spacing function.

        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        assert ray_bundle is not None
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_rays = ray_bundle.origins.shape[0]

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]  # [1, num_samples+1]

        # TODO More complicated than it needs to be.
        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand

        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.nears.clone(), ray_bundle.fars.clone()))
        spacing_to_euclidean_fn = lambda x: self.spacing_fn_inv(x * s_far + (1 - x) * s_near)
        euclidean_bins = spacing_to_euclidean_fn(bins)  # [num_rays, num_samples+1]

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )

        return ray_samples


class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LinearDisparitySampler(SpacedSampler):
    """Sample linearly in disparity along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: 1 / x,
            spacing_fn_inv=lambda x: 1 / x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class SqrtSampler(SpacedSampler):
    """Square root sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.sqrt,
            spacing_fn_inv=lambda x: x**2,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LogSampler(SpacedSampler):
    """Log sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.log,
            spacing_fn_inv=torch.exp,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class UniformLinDispPiecewiseSampler(SpacedSampler):
    """Piecewise sampler along a ray that allocates the first half of the samples uniformly and the second half
    using linearly in disparity spacing.


    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class PDFSampler(Sampler):
    """Sample based on probability distribution

    Args:
        num_samples: Number of samples per ray
        train_stratified: Randomize location within each bin during training.
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
        include_original: Add original samples to ray.
        histogram_padding: Amount to weights prior to computing PDF.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified: bool = True,
        single_jitter: bool = False,
        include_original: bool = True,
        histogram_padding: float = 0.01,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.include_original = include_original
        self.histogram_padding = histogram_padding
        self.single_jitter = single_jitter

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        ray_samples: Optional[RaySamples] = None,
        weights: TensorType[..., "num_samples", 1] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin
            num_samples: Number of samples per ray
            eps: Small value to prevent numerical issues.

        Returns:
            Positions and deltas for samples along a ray
        """

        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples and ray_bundle must be provided")

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_bins = num_samples + 1

        weights = weights[..., 0] + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.train_stratified and self.training:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert (
            ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert ray_samples.spacing_to_euclidean_fn is not None, "ray_samples.spacing_to_euclidean_fn must be provided"
        existing_bins = torch.cat(
            [
                ray_samples.spacing_starts[..., 0],
                ray_samples.spacing_ends[..., -1:, 0],
            ],
            dim=-1,
        )

        inds = torch.searchsorted(cdf, u, side="right")
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )

        return ray_samples


class VolumetricSampler(Sampler):
    """Sampler inspired by the one proposed in the Instant-NGP paper.
    Generates samples along a ray by sampling the occupancy field.
    Optionally removes occluded samples if the density_fn is provided.
    Args:
    occupancy_grid: Occupancy grid to sample from.
    density_fn: Function that evaluates density at a given point.
    scene_aabb: Axis-aligned bounding box of the scene, should be set to None if the scene is unbounded.
    """

    def __init__(
        self,
        occupancy_grid: Optional[OccupancyGrid] = None,
        density_fn: Optional[Callable[[TensorType[..., 3]], TensorType[..., 1]]] = None,
        scene_aabb: Optional[TensorType[2, 3]] = None,
    ) -> None:

        super().__init__()
        self.scene_aabb = scene_aabb
        self.density_fn = density_fn
        self.occupancy_grid = occupancy_grid
        if self.scene_aabb is not None:
            self.scene_aabb = self.scene_aabb.to("cuda").flatten()
        print(self.scene_aabb)

    def get_sigma_fn(self, origins, directions) -> Optional[Callable]:
        """Returns a function that returns the density of a point.
        Args:
            origins: Origins of rays
            directions: Directions of rays
        Returns:
            Function that returns the density of a point or None if a density function is not provided.
        """

        if self.density_fn is None or not self.training:
            return None

        density_fn = self.density_fn

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            return density_fn(positions)

        return sigma_fn

    def generate_ray_samples(self) -> RaySamples:
        raise RuntimeError(
            "The VolumetricSampler fuses sample generation and density check together. Please call forward() directly."
        )

    # pylint: disable=arguments-differ
    def forward(
        self,
        ray_bundle: RayBundle,
        render_step_size: float,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
        cone_angle: float = 0.0,
        alpha_thre: float = 1e-2,
    ) -> Tuple[RaySamples, TensorType["total_samples",]]:
        """Generate ray samples in a bounding box.
        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.
            alpha_thre: Threshold for ray marching
        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None

        ray_indices, starts, ends = nerfacc.ray_marching(
            rays_o=rays_o,
            rays_d=rays_d,
            scene_aabb=self.scene_aabb,
            grid=self.occupancy_grid,
            # this is a workaround - using density causes crash and damage quality. should be fixed
            sigma_fn=None,  # self.get_sigma_fn(rays_o, rays_d),
            render_step_size=render_step_size,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=self.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1, 1), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1, 1), dtype=ends.dtype, device=rays_o.device)

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]

        zeros = torch.zeros_like(origins[:, :1])
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts,
                ends=ends,
                pixel_area=zeros,
            ),
            camera_indices=camera_indices,
        )
        return ray_samples, ray_indices


class ProposalNetworkSampler(Sampler):
    """Sampler that uses a proposal network to generate samples."""

    def __init__(
        self,
        num_proposal_samples_per_ray: Tuple[int, ...] = (64,),
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 2,
        use_uniform_sampler: bool = False,
        single_jitter: bool = False,
        update_sched: Callable = lambda x: 1,
    ) -> None:
        super().__init__()
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")

        if use_uniform_sampler:
            self.initial_sampler = UniformSampler(single_jitter=single_jitter)
        else:
            self.initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)

        self.pdf_sampler = PDFSampler(include_original=False, single_jitter=single_jitter)

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def set_anneal(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._anneal = anneal

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self._steps_since_update += 1

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](ray_samples.frustums.get_positions())
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.frustums.get_positions())
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list


class ErrorBoundedSampler(Sampler):
    """VolSDF's error bounded sampler that uses a sdf network to generate samples."""

    def __init__(
        self,
        num_samples: int = 64,
        num_samples_eval: int = 128,
        num_samples_extra: int = 32,
        eps: float = 0.1,
        beta_iters: int = 10,
        max_total_iters: int = 5,
        add_tiny: float = 1e-6,
        single_jitter: bool = False,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_samples_eval = num_samples_eval
        self.num_samples_extra = num_samples_extra
        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.add_tiny = add_tiny
        self.single_jitter = single_jitter

        # samplers
        self.uniform_sampler = UniformSampler(single_jitter=single_jitter)
        self.pdf_sampler = PDFSampler(
            include_original=False,
            single_jitter=single_jitter,
            histogram_padding=1e-5,
        )

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fn: Optional[Callable] = None,
        sdf_fn: Optional[Callable] = None,
        return_eikonal_points: bool = True,
    ) -> Union[Tuple[RaySamples, torch.Tensor], RaySamples]:
        assert ray_bundle is not None
        assert density_fn is not None
        assert sdf_fn is not None

        beta0 = density_fn.get_beta().detach()

        # Start with uniform sampling
        ray_samples = self.uniform_sampler(ray_bundle, num_samples=self.num_samples_eval)

        # Get maximum beta from the upper bound (Lemma 2)
        deltas = ray_samples.deltas.squeeze(-1)

        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (deltas**2.0).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True
        sorted_index = None
        new_samples = ray_samples

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:

            with torch.no_grad():
                new_sdf = sdf_fn(new_samples)

            # merge sdf predictions
            if sorted_index is not None:
                sdf_merge = torch.cat([sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
                sdf = torch.gather(sdf_merge, 1, sorted_index).unsqueeze(-1)
            else:
                sdf = new_sdf

            # Calculating the bound d* (Theorem 1)
            d_star = self.get_dstar(sdf, ray_samples)

            # Updating beta using line search
            beta = self.get_updated_beta(beta0, beta, density_fn, sdf, d_star, ray_samples)

            # Upsample more points
            density = density_fn(sdf.reshape(ray_samples.shape), beta=beta.unsqueeze(-1))

            weights, transmittance = ray_samples.get_weights_and_transmittance(density.unsqueeze(-1))

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                # Sample more points proportional to the current error bound
                deltas = ray_samples.deltas.squeeze(-1)

                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1)) * (deltas**2.0) / (4 * beta.unsqueeze(-1) ** 2)
                )

                error_integral = torch.cumsum(error_per_section, dim=-1)
                weights = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * transmittance[..., 0]

                new_samples = self.pdf_sampler(
                    ray_bundle, ray_samples, weights.unsqueeze(-1), num_samples=self.num_samples_eval
                )

                ray_samples, sorted_index = self.merge_ray_samples(ray_bundle, ray_samples, new_samples)

            else:
                # Sample the final sample set to be used in the volume rendering integral
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, weights, num_samples=self.num_samples)

        if return_eikonal_points:
            # sample some of the near surface points for eikonal loss
            sampled_points = ray_samples.frustums.get_positions().view(-1, 3)
            idx = torch.randint(sampled_points.shape[0], (ray_samples.shape[0] * 10,)).to(sampled_points.device)
            points = sampled_points[idx]

        # Add extra samples uniformly
        if self.num_samples_extra > 0:
            ray_samples_uniform = self.uniform_sampler(ray_bundle, num_samples=self.num_samples_extra)
            ray_samples, _ = self.merge_ray_samples(ray_bundle, ray_samples, ray_samples_uniform)

        if return_eikonal_points:
            return ray_samples, points

        return ray_samples

    def get_dstar(self, sdf, ray_samples: RaySamples):
        """Calculating the bound d* (Theorem 1) from VolSDF"""
        d = sdf.reshape(ray_samples.shape)
        dists = ray_samples.deltas.squeeze(-1)
        a, b, c = dists[:, :-1], d[:, :-1].abs(), d[:, 1:].abs()
        first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
        second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
        d_star = torch.zeros(ray_samples.shape[0], ray_samples.shape[1] - 1).to(d.device)
        d_star[first_cond] = b[first_cond]
        d_star[second_cond] = c[second_cond]
        s = (a + b + c) / 2.0
        area_before_sqrt = s * (s - a) * (s - b) * (s - c)
        mask = ~first_cond & ~second_cond & (b + c - a > 0)
        d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
        d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

        # padding to make the same shape as ray_samples
        # d_star_left = torch.cat((d_star[:, :1], d_star), dim=-1)
        # d_star_right = torch.cat((d_star, d_star[:, -1:]), dim=-1)
        # d_star = torch.minimum(d_star_left, d_star_right)

        d_star = torch.cat((d_star, d_star[:, -1:]), dim=-1)
        return d_star

    def get_updated_beta(self, beta0, beta, density_fn, sdf, d_star, ray_samples: RaySamples):
        curr_error = self.get_error_bound(beta0, density_fn, sdf, d_star, ray_samples)
        beta[curr_error <= self.eps] = beta0
        beta_min, beta_max = beta0.repeat(ray_samples.shape[0]), beta
        for j in range(self.beta_iters):
            beta_mid = (beta_min + beta_max) / 2.0
            curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), density_fn, sdf, d_star, ray_samples)
            beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
            beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
        beta = beta_max
        return beta

    def get_error_bound(self, beta, density_fn, sdf, d_star, ray_samples):
        """Get error bound from VolSDF"""
        densities = density_fn(sdf.reshape(ray_samples.shape), beta=beta)

        deltas = ray_samples.deltas.squeeze(-1)
        delta_density = deltas * densities

        integral_estimation = torch.cumsum(delta_density[..., :-1], dim=-1)
        integral_estimation = torch.cat(
            [torch.zeros((*integral_estimation.shape[:1], 1), device=densities.device), integral_estimation], dim=-1
        )

        error_per_section = torch.exp(-d_star / beta) * (deltas**2.0) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * torch.exp(-integral_estimation)

        return bound_opacity.max(-1)[0]

    def merge_ray_samples(self, ray_bundle: RayBundle, ray_samples_1: RaySamples, ray_samples_2: RaySamples):
        """Merge two set of ray samples and return sorted index which can be used to merge sdf values

        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
        """

        starts_1 = ray_samples_1.spacing_starts[..., 0]
        starts_2 = ray_samples_2.spacing_starts[..., 0]

        ends = torch.maximum(ray_samples_1.spacing_ends[..., -1:, 0], ray_samples_2.spacing_ends[..., -1:, 0])

        bins, sorted_index = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

        bins = torch.cat([bins, ends], dim=-1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples_1.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples_1.spacing_to_euclidean_fn,
        )

        return ray_samples, sorted_index


def save_points(path_save, pts, colors=None, normals=None, BRG2RGB=False):
    """save points to point cloud using open3d"""
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3
    import numpy as np
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(path_save, cloud)


class NeuSSampler(Sampler):
    """NeuS sampler that uses a sdf network to generate samples with fixed variance value in each iterations."""

    def __init__(
        self,
        num_samples: int = 64,
        num_samples_importance: int = 64,
        num_samples_outside: int = 32,
        num_upsample_steps: int = 4,
        base_variance: float = 64,
        single_jitter: bool = True,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_samples_importance = num_samples_importance
        self.num_samples_outside = num_samples_outside
        self.num_upsample_steps = num_upsample_steps
        self.base_variance = base_variance
        self.single_jitter = single_jitter

        # samplers
        self.uniform_sampler = UniformSampler(single_jitter=single_jitter)
        self.pdf_sampler = PDFSampler(
            include_original=False,
            single_jitter=single_jitter,
            histogram_padding=1e-5,
        )
        self.outside_sampler = LinearDisparitySampler()
        # TODO make it outside
        # for merge samples
        self.error_bounded_sampler = ErrorBoundedSampler()

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        sdf_fn: Optional[Callable] = None,
        ray_samples: Optional[RaySamples] = None,
    ) -> Union[Tuple[RaySamples, torch.Tensor], RaySamples]:
        assert ray_bundle is not None
        assert sdf_fn is not None

        # Start with uniform sampling
        if ray_samples is None:
            ray_samples = self.uniform_sampler(ray_bundle, num_samples=self.num_samples)

        total_iters = 0
        sorted_index = None
        new_samples = ray_samples

        base_variance = self.base_variance

        while total_iters < self.num_upsample_steps:

            with torch.no_grad():
                new_sdf = sdf_fn(new_samples)

            # merge sdf predictions
            if sorted_index is not None:
                sdf_merge = torch.cat([sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
                sdf = torch.gather(sdf_merge, 1, sorted_index).unsqueeze(-1)
            else:
                sdf = new_sdf

            # compute with fix variances
            alphas = self.rendering_sdf_with_fixed_inv_s(
                ray_samples, sdf.reshape(ray_samples.shape), inv_s=base_variance * 2**total_iters
            )

            weights = ray_samples.get_weights_from_alphas(alphas[..., None])
            weights = torch.cat((weights, torch.zeros_like(weights[:, :1])), dim=1)

            new_samples = self.pdf_sampler(
                ray_bundle,
                ray_samples,
                weights,
                num_samples=self.num_samples_importance // self.num_upsample_steps,
            )

            ray_samples, sorted_index = self.error_bounded_sampler.merge_ray_samples(
                ray_bundle, ray_samples, new_samples
            )

            total_iters += 1

        # save_points("p.ply", ray_samples.frustums.get_start_positions().detach().cpu().numpy().reshape(-1, 3))
        # exit(-1)
        # TODO
        # sample more points outside surface
        # if self.num_samples_outside > 0:
        #   ray_samples_uniform = self.outside_sampler(ray_bundle, num_samples=self.num_samples_outside)
        #     ray_samples, _ = self.merge_ray_samples(ray_bundle, ray_samples, ray_samples_uniform)

        return ray_samples

    def rendering_sdf_with_fixed_inv_s(self, ray_samples: RaySamples, sdf: torch.Tensor, inv_s):
        """rendering given a fixed inv_s as NeuS"""
        batch_size = ray_samples.shape[0]
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        deltas = ray_samples.deltas[:, :-1, 0]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (deltas + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0)

        dist = deltas
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        return alpha


class UniSurfSampler(Sampler):
    """NeuS sampler that uses a sdf network to generate samples with fixed variance value in each iterations."""

    def __init__(
        self,
        num_samples_interval: int = 64,
        num_samples_outside: int = 32,
        num_samples_importance: int = 32,
        num_marching_steps: int = 256,
        num_secant_steps: int = 8,
        interval_start: float = 0.25,
        interval_end: float = 0.0125,
        interval_decay: float = 0.00005,  # default value is 0.000015 and will reach the end value at 200K while 0.00005 will reach the end at 60K iter
        single_jitter: bool = False,
    ) -> None:
        super().__init__()
        self.num_samples_interval = num_samples_interval
        self.num_samples_outside = num_samples_outside
        self.num_samples_importance = num_samples_importance
        self.num_marching_steps = num_marching_steps
        self.num_secant_steps = num_secant_steps
        self.interval_start = interval_start
        self.interval_end = interval_end
        self.interval_decay = interval_decay
        self.single_jitter = single_jitter

        # samplers
        self.uniform_sampler = UniformSampler(single_jitter=single_jitter)
        self.outside_sampler = UniformSampler(single_jitter=single_jitter)
        self.pdf_sampler = PDFSampler(
            include_original=False,
            single_jitter=single_jitter,
            histogram_padding=1e-5,
        )
        # for merge samples
        self.error_bounded_sampler = ErrorBoundedSampler()

        # step counter
        self._step = 0
        self.delta = self.interval_start

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self.delta = max(self.interval_start * math.exp(-1 * self.interval_decay * self._step), self.interval_end)

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        occupancy_fn: Optional[RayBundle] = None,
        sdf_fn: Optional[Callable] = None,
        return_surface_points: bool = False,
    ) -> Union[Tuple[RaySamples, torch.Tensor], RaySamples]:
        assert ray_bundle is not None
        assert sdf_fn is not None

        # Start with uniform sampling
        ray_samples = self.uniform_sampler(ray_bundle, num_samples=self.num_marching_steps)

        with torch.no_grad():
            sdf = sdf_fn(ray_samples)

        # importance sampling
        occupancy = occupancy_fn(sdf)
        weights = ray_samples.get_weights_from_alphas(occupancy)

        importance_samples = self.pdf_sampler(
            ray_bundle,
            ray_samples,
            weights,
            num_samples=self.num_samples_importance,
        )

        # samples uniformly with near and far
        ray_samples_uniform_outside = self.outside_sampler(ray_bundle, num_samples=self.num_samples_outside)

        # merge
        ray_samples_uniform_importance, _ = self.error_bounded_sampler.merge_ray_samples(
            ray_bundle, importance_samples, ray_samples_uniform_outside
        )

        # surface points
        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        n_rays, n_samples = ray_samples.shape
        starts = ray_samples.frustums.starts
        sign_matrix = torch.cat(
            [torch.sign(sdf[:, :-1, 0] * sdf[:, 1:, 0]), torch.ones(n_rays, 1).to(sdf.device)], dim=-1
        )
        cost_matrix = sign_matrix * torch.arange(n_samples, 0, -1).float().to(sdf.device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_pos_to_neg = sdf[torch.arange(n_rays), indices, 0] > 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_pos_to_neg

        # Get depth values and function values for the interval
        d_low = starts[torch.arange(n_rays), indices, 0][mask]
        v_low = sdf[torch.arange(n_rays), indices, 0][mask]

        indices = torch.clamp(indices + 1, max=n_samples - 1)
        d_high = starts[torch.arange(n_rays), indices, 0][mask]
        v_high = sdf[torch.arange(n_rays), indices, 0][mask]

        # TODO secant method
        # linear-interpolations
        z = (v_low * d_high - v_high * d_low) / (v_low - v_high)

        # make this simpler
        origins = ray_samples.frustums.origins[torch.arange(n_rays), indices, :][mask]
        directions = ray_samples.frustums.directions[torch.arange(n_rays), indices, :][mask]
        surface_points = origins + directions * z[..., None]

        if surface_points.shape[0] <= 0:
            surface_points = torch.rand((1024, 3), device=sdf.device) - 0.5

        # modify near and far values according current schedule
        nears, fars = ray_bundle.nears.clone(), ray_bundle.fars.clone()
        dists = fars - nears

        ray_bundle.nears[mask] = z[:, None] - dists[mask] * self.delta
        ray_bundle.fars[mask] = z[:, None] + dists[mask] * self.delta
        # min max bound
        ray_bundle.nears = torch.maximum(ray_bundle.nears, nears)
        ray_bundle.fars = torch.minimum(ray_bundle.fars, fars)

        # samples uniformly with new surface interval
        ray_samples_interval = self.uniform_sampler(ray_bundle, num_samples=self.num_samples_interval)

        # change back to original values
        ray_bundle.nears = nears
        ray_bundle.fars = fars

        # merge sampled points
        ray_samples = self.merge_ray_samples_in_eculidean(
            ray_bundle, ray_samples_interval, ray_samples_uniform_importance
        )

        if return_surface_points:
            return ray_samples, surface_points

        return ray_samples

    def merge_ray_samples_in_eculidean(
        self, ray_bundle: RayBundle, ray_samples_1: RaySamples, ray_samples_2: RaySamples
    ):
        """Merge two set of ray samples and return sorted index which can be used to merge sdf values

        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
        """
        starts_1 = ray_samples_1.spacing_to_euclidean_fn(ray_samples_1.spacing_starts[..., 0])
        starts_2 = ray_samples_2.spacing_to_euclidean_fn(ray_samples_2.spacing_starts[..., 0])

        end_1 = ray_samples_1.spacing_to_euclidean_fn(ray_samples_1.spacing_ends[:, -1:, 0])
        end_2 = ray_samples_2.spacing_to_euclidean_fn(ray_samples_2.spacing_ends[:, -1:, 0])

        end = torch.maximum(end_1, end_2)

        euclidean_bins, _ = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

        euclidean_bins = torch.cat([euclidean_bins, end], dim=-1)

        # Stop gradients
        euclidean_bins = euclidean_bins.detach()

        # TODO convert euclidean bins to spacing bins
        bins = euclidean_bins

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples_1.spacing_to_euclidean_fn,
        )

        return ray_samples

    def secant_method(
        self,
        ray_bundle: Optional[RayBundle] = None,
        sdf_fn: Optional[Callable] = None,
    ):
        """run secant method to refine surface points"""
        raise NotImplementedError


class NeuralReconWSampler(Sampler):
    """Voxel surface guided sampler in NeuralReconW."""

    def __init__(
        self,
        aabb,
        coarse_binary_grid,
        coarse_resolution: int = 32,
        fine_resolution: int = 512,
        num_samples: int = 8,
        num_samples_importance: int = 16,
        num_samples_boundary: int = 10,
        steps_per_grid_update: int = 5000,
        local_rank: int = 0,
        single_jitter: bool = False,
    ) -> None:
        super().__init__()
        self.aabb = aabb
        self.coarse_resolution = coarse_resolution
        self.fine_resolution = fine_resolution
        self.num_samples = num_samples
        self.num_samples_importance = num_samples_importance
        self.num_samples_boundary = num_samples_boundary
        self.single_jitter = single_jitter
        self.steps_per_grid_update = steps_per_grid_update
        self.local_rank = local_rank

        # TODO remvoe 2.0 and create cube in the initialization
        self.grid_size = self.coarse_resolution
        self.voxel_size = 2.0 / self.grid_size
        self.fine_grid_size = self.fine_resolution // self.grid_size

        # for voxel guided sampling
        self.uniform_sampler = UniformSampler(single_jitter=False)

        # for surface guided sampling
        self.neus_sampler = NeuSSampler(
            num_samples=num_samples,
            num_samples_importance=num_samples_importance,
            num_samples_outside=0,
            num_upsample_steps=2,
            base_variance=512,
        )

        # for merge samples
        self.unisurf_sampler = UniSurfSampler()

        self.grid = nerfacc.OccupancyGrid(aabb.reshape(-1), resolution=self.coarse_resolution)
        self._binary = coarse_binary_grid.reshape(
            self.coarse_resolution, self.coarse_resolution, self.coarse_resolution
        ).contiguous()
        self._binary_fine = None

        self.init_grid_coordinate()

    def init_grid_coordinate(self):
        # fine grid coordinates in each coarse voxel
        offset = torch.linspace(-1.0, 1.0, self.fine_grid_size * 2 + 1)[1::2]
        x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
        fine_offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3) * self.voxel_size * 0.5

        # coarse grid coordinates
        offset = torch.linspace(-1.0 + self.voxel_size / 2.0, 1.0 - self.voxel_size / 2.0, self.grid_size)
        x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
        coarse_offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        self.register_buffer("coarse_offset_cube", coarse_offset_cube)
        self.register_buffer("fine_offset_cube", fine_offset_cube)

    @torch.no_grad()
    def update_binary_grid(self, step, sdf_fn=None):
        assert sdf_fn is not None
        # bootstrap should needs longer if using only one gpus
        if step >= self.steps_per_grid_update and step % self.steps_per_grid_update == 0:
            device = self.coarse_offset_cube.device

            mask = torch.zeros((self.grid_size**3, self.fine_grid_size**3), dtype=torch.bool, device=device)

            occupied_voxel = self.coarse_offset_cube[self._binary.reshape(-1)]
            fine_voxel = occupied_voxel[:, None] + self.fine_offset_cube[None, :]

            fine_voxel = fine_voxel.reshape(-1, 3)
            # save_points("fine_voxel.ply", fine_voxel.cpu().numpy())

            def evaluate(points):
                z = []
                for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                    z.append(sdf_fn(pnts))
                z = torch.cat(z, axis=0)
                return z

            sdf = evaluate(fine_voxel)
            sdf = sdf.reshape(occupied_voxel.shape[0], self.fine_grid_size**3)
            sdf_mask = sdf <= 0.0
            mask[self._binary.reshape(-1)] = sdf_mask

            self._binary_fine = (
                mask.reshape([self.grid_size] * 3 + [self.fine_grid_size] * 3)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(self.fine_resolution, self.fine_resolution, self.fine_resolution)
                .contiguous()
            )

            # offset = torch.linspace(-1.0, 1.0, self.fine_resolution * 2 + 1, device=device)[1::2]
            # x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
            # grid_coord = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
            # save_points("fine_voxel_valid.ply", grid_coord[self._binary_fine.reshape(-1)].cpu().numpy())

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        sdf_fn: Optional[Callable] = None,
    ) -> Union[Tuple[RaySamples, torch.Tensor], RaySamples]:
        assert ray_bundle is not None
        assert sdf_fn is not None

        # near and far from occupancy grids
        packed_info, _, t_starts, t_ends = nerfacc.cuda.ray_marching(
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
        tt_ends = nerfacc.unpack_data(packed_info, t_ends)

        hit_grid = (tt_starts > 0).any(dim=1)[:, 0]
        if hit_grid.float().sum() > 0:
            ray_bundle.nears[hit_grid] = tt_starts[hit_grid][:, 0]
            ray_bundle.fars[hit_grid] = tt_ends[hit_grid].max(dim=1)[0]

        # sample uniformly with currently nears and far
        voxel_samples = self.uniform_sampler(ray_bundle, num_samples=self.num_samples_boundary)

        if self._binary_fine is not None:
            # near from finer occupancy grids
            packed_info, _, t_starts, t_ends = nerfacc.cuda.ray_marching(
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

            # update with near and far
            hit_grid = (tt_starts > 0).any(dim=1)[:, 0]
            if hit_grid.float().sum() > 0:
                ray_bundle.nears[hit_grid] = tt_starts[hit_grid][:, 0] - 0.03
                ray_bundle.fars[hit_grid] = tt_starts[hit_grid][:, 0] + 0.03
            else:
                print("waring not intersection")
            print("sampling around surfaces")

        # surface guided sampling
        surface_samples = self.neus_sampler(ray_bundle, sdf_fn=sdf_fn)

        # merge samples
        ray_samples = self.unisurf_sampler.merge_ray_samples_in_eculidean(ray_bundle, surface_samples, voxel_samples)

        return ray_samples


class NeuSAccSampler(Sampler):
    """Voxel surface guided sampler in NeuralReconW."""

    def __init__(
        self,
        aabb,
        neus_sampler: NeuSSampler = None,
        resolution: int = 128,
        num_samples: int = 8,
        num_samples_importance: int = 16,
        num_samples_boundary: int = 10,
        steps_warpup: int = 2000,
        steps_per_grid_update: int = 1000,
        importance_sampling: bool = False,
        local_rank: int = 0,
        single_jitter: bool = False,
    ) -> None:
        super().__init__()
        self.aabb = aabb
        self.resolution = resolution
        self.num_samples = num_samples
        self.num_samples_importance = num_samples_importance
        self.num_samples_boundary = num_samples_boundary
        self.single_jitter = single_jitter
        self.importance_sampling = importance_sampling
        self.steps_warpup = steps_warpup
        self.steps_per_grid_update = steps_per_grid_update
        self.local_rank = local_rank
        self.step_size = 0.01 / 5.0
        self.alpha_thres = 0.001

        # only supports cubic bbox for now
        assert aabb[0, 0] == aabb[0, 1] and aabb[0, 0] == aabb[0, 2]
        assert aabb[1, 0] == aabb[1, 1] and aabb[1, 0] == aabb[1, 2]
        self.grid_size = self.resolution
        self.voxel_size = (aabb[1, 0] - aabb[0, 0]) / self.grid_size

        # nesu_sampler at the begining of training
        self.neus_sampler = neus_sampler

        self.grid = nerfacc.OccupancyGrid(aabb.reshape(-1), resolution=self.resolution)
        self.register_buffer("_binary", torch.ones((self.grid_size, self.grid_size, self.grid_size), dtype=torch.bool))
        self.register_buffer("_update_counter", torch.zeros(1, dtype=torch.int32))

        self.init_grid_coordinate()

    def init_grid_coordinate(self):
        # coarse grid coordinates
        aabb = self.aabb
        offset_x = torch.linspace(
            aabb[0, 0] + self.voxel_size / 2.0, aabb[1, 0] - self.voxel_size / 2.0, self.grid_size
        )
        offset_y = torch.linspace(
            aabb[0, 1] + self.voxel_size / 2.0, aabb[1, 1] - self.voxel_size / 2.0, self.grid_size
        )
        offset_z = torch.linspace(
            aabb[0, 2] + self.voxel_size / 2.0, aabb[1, 2] - self.voxel_size / 2.0, self.grid_size
        )
        x, y, z = torch.meshgrid(offset_x, offset_y, offset_z, indexing="ij")
        cube_coordinate = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        self.register_buffer("cube_coordinate", cube_coordinate)

    def update_step_size(self, step, inv_s=None):
        assert inv_s is not None
        inv_s = inv_s().item()
        self.step_size = 14.0 / inv_s / 16

    @torch.no_grad()
    def update_binary_grid(self, step, sdf_fn=None, inv_s=None):
        assert sdf_fn is not None
        assert inv_s is not None

        if step >= self.steps_warpup and step % self.steps_per_grid_update == 0:

            mask = self._binary.reshape(-1)
            # TODO voxels can't be recovered once it is pruned
            occupied_voxel = self.cube_coordinate[mask.reshape(-1)]

            # save_points("occupied_voxel.ply", occupied_voxel.cpu().numpy())

            def evaluate(points):
                z = []
                for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                    z.append(sdf_fn(pnts))
                z = torch.cat(z, axis=0)
                return z

            sdf = evaluate(occupied_voxel)

            # sdf_mask = sdf.abs() <= 0.04

            # use maximum bound for sdf value
            bound = self.voxel_size * (3**0.5) / 2.0
            sdf = sdf.abs()
            sdf = torch.maximum(sdf - bound, torch.zeros_like(sdf))

            estimated_next_sdf = sdf - self.step_size * 0.5
            estimated_prev_sdf = sdf + self.step_size * 0.5
            inv_s = inv_s()
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
            sdf_mask = alpha > self.alpha_thres

            mask[mask.reshape(-1).clone()] = sdf_mask

            self._binary = mask.reshape([self.grid_size] * 3).contiguous()

            # save_points("voxel_valid.ply", self.cube_coordinate[self._binary.reshape(-1)].cpu().numpy())

            # TODO do we need dilation
            # F.max_pool3d(M.float(), kernel_size=3, padding=1, stride=1).bool()
            # save_points("voxel_valid_dilated.ply", self.cube_coordinate[self._binary.reshape(-1)].cpu().numpy())
            self._update_counter += 1

    def create_ray_samples_from_ray_indices(self, ray_bundle: RayBundle, ray_indices, t_starts, t_ends):
        rays_o = ray_bundle.origins[ray_indices]
        rays_d = ray_bundle.directions[ray_indices]
        camera_indices = ray_bundle.camera_indices[ray_indices]
        deltas = t_ends - t_starts

        frustums = Frustums(
            origins=rays_o,  # [..., 1, 3]
            directions=rays_d,  # [..., 1, 3]
            starts=t_starts,  # [..., num_samples, 1]
            ends=t_ends,  # [..., num_samples, 1]
            pixel_area=torch.ones_like(t_starts),  # [..., 1, 1]
        )

        ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1]
        )
        return ray_samples

    @torch.no_grad()
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        sdf_fn: Optional[Callable] = None,
        alpha_fn: Optional[Callable] = None,
    ) -> Union[Tuple[RaySamples, torch.Tensor], RaySamples]:
        assert ray_bundle is not None
        assert sdf_fn is not None

        # sampler with original neus Sampler?
        if self._update_counter.item() <= 0:
            return self.neus_sampler(ray_bundle, sdf_fn=sdf_fn)

        assert alpha_fn is not None

        # sampling from occupancy grids
        packed_info, ray_indices, t_starts, t_ends = nerfacc.cuda.ray_marching(
            ray_bundle.origins.contiguous(),
            ray_bundle.directions.contiguous(),
            ray_bundle.nears[:, 0].contiguous(),
            ray_bundle.fars[:, 0].contiguous(),
            self.grid.roi_aabb.contiguous(),
            self._binary,
            self.grid.contraction_type.to_cpp_version(),
            self.step_size,  # TODO stepsize based on inv_s value?
            0.0,
        )

        # create ray_samples with the intersection
        ray_indices = ray_indices.long()
        ray_samples = self.create_ray_samples_from_ray_indices(ray_bundle, ray_indices, t_starts, t_ends)

        if self.importance_sampling and ray_samples.shape[0] > 0:
            # save_points("first.ply", ray_samples.frustums.get_start_positions().cpu().numpy().reshape(-1, 3))

            alphas = alpha_fn(ray_samples)
            weights = nerfacc.render_weight_from_alpha(alphas, ray_indices=ray_indices, n_rays=ray_bundle.shape[0])

            # TODO make it configurable
            # re sample
            packed_info, t_starts, t_ends = nerfacc.ray_resampling(packed_info, t_starts, t_ends, weights[:, 0], 16)
            ray_indices = nerfacc.unpack_info(packed_info, t_starts.shape[0])
            ray_samples = self.create_ray_samples_from_ray_indices(ray_bundle, ray_indices, t_starts, t_ends)

            # save_points("second.ply", ray_samples.frustums.get_start_positions().cpu().numpy().reshape(-1, 3))

        return ray_samples, ray_indices
