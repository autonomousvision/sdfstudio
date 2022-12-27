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
Collection of Losses.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1.0e-7


def outer(
    t0_starts: TensorType[..., "num_samples_0"],
    t0_ends: TensorType[..., "num_samples_0"],
    t1_starts: TensorType[..., "num_samples_1"],
    t1_ends: TensorType[..., "num_samples_1"],
    y1: TensorType[..., "num_samples_1"],
) -> TensorType[..., "num_samples_0"]:
    """Faster version of

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right") - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def lossfun_outer(
    t: TensorType[..., "num_samples+1"],
    w: TensorType[..., "num_samples"],
    t_env: TensorType[..., "num_samples+1"],
    w_env: TensorType[..., "num_samples"],
):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping historgram
        w_env: weights that should upper bound the inner (t,w) histogram
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s space"""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
    return sdist


def interlevel_loss(weights_list, ray_samples_list):
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))
    return loss_interlevel


# Verified
def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def distortion_loss(weights_list, ray_samples_list):
    """From mipnerf360"""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss


def nerfstudio_distortion_loss(
    ray_samples: RaySamples,
    densities: TensorType["bs":..., "num_samples", 1] = None,
    weights: TensorType["bs":..., "num_samples", 1] = None,
) -> TensorType["bs":..., 1]:
    """Ray based distortion loss proposed in MipNeRF-360. Returns distortion Loss.

    .. math::

        \\mathcal{L}(\\mathbf{s}, \\mathbf{w}) =\\iint\\limits_{-\\infty}^{\\,\\,\\,\\infty}
        \\mathbf{w}_\\mathbf{s}(u)\\mathbf{w}_\\mathbf{s}(v)|u - v|\\,d_{u}\\,d_{v}

    where :math:`\\mathbf{w}_\\mathbf{s}(u)=\\sum_i w_i \\mathbb{1}_{[\\mathbf{s}_i, \\mathbf{s}_{i+1})}(u)`
    is the weight at location :math:`u` between bin locations :math:`s_i` and :math:`s_{i+1}`.

    Args:
        ray_samples: Ray samples to compute loss over
        densities: Predicted sample densities
        weights: Predicted weights from densities and sample locations
    """
    if torch.is_tensor(densities):
        assert not torch.is_tensor(weights), "Cannot use both densities and weights"
        # Compute the weight at each sample location
        weights = ray_samples.get_weights(densities)
    if torch.is_tensor(weights):
        assert not torch.is_tensor(densities), "Cannot use both densities and weights"

    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends

    assert starts is not None and ends is not None, "Ray samples must have spacing starts and ends"
    midpoints = (starts + ends) / 2.0  # (..., num_samples, 1)

    loss = (
        weights * weights[..., None, :, 0] * torch.abs(midpoints - midpoints[..., None, :, 0])
    )  # (..., num_samples, num_samples)
    loss = torch.sum(loss, dim=(-1, -2))[..., None]  # (..., num_samples)
    loss = loss + 1 / 3.0 * torch.sum(weights**2 * (ends - starts), dim=-2)

    return loss


def orientation_loss(
    weights: TensorType["bs":..., "num_samples", 1],
    normals: TensorType["bs":..., "num_samples", 3],
    viewdirs: TensorType["bs":..., 3],
):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    """
    w = weights
    n = normals
    v = viewdirs
    n_dot_v = (n * v[..., None, :]).sum(axis=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)


def pred_normal_loss(
    weights: TensorType["bs":..., "num_samples", 1],
    normals: TensorType["bs":..., "num_samples", 3],
    pred_normals: TensorType["bs":..., "num_samples", 3],
):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)


def monosdf_normal_loss(normal_pred: torch.Tensor, normal_gt: torch.Tensor):
    """normal consistency loss as monosdf

    Args:
        normal_pred (torch.Tensor): volume rendered normal
        normal_gt (torch.Tensor): monocular normal
    """
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
    cos = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
    return l1 + cos


# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MiDaSMSELoss(nn.Module):
    def __init__(self, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
            )

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super().__init__()

        self.__data_loss = MiDaSMSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


# end copy


# copy from https://github.com/svip-lab/Indoor-SfMLearner/blob/0d682b7ce292484e5e3e2161fc9fc07e2f5ca8d1/layers.py#L218
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self, patch_size):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(patch_size, 1)
        self.mu_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_x_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(patch_size, 1)

        self.refl = nn.ReflectionPad2d(patch_size // 2)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


# TODO test different losses
class NCC(nn.Module):
    """Layer to compute the normalization cross correlation (NCC) of patches"""

    def __init__(self, patch_size: int = 11, min_patch_variance: float = 0.01):
        super(NCC, self).__init__()
        self.patch_size = patch_size
        self.min_patch_variance = min_patch_variance

    def forward(self, x, y):
        # TODO if we use gray image we should do it right after loading the image to save computations
        # to gray image
        x = torch.mean(x, dim=1)
        y = torch.mean(y, dim=1)

        x_mean = torch.mean(x, dim=(1, 2), keepdim=True)
        y_mean = torch.mean(y, dim=(1, 2), keepdim=True)

        x_normalized = x - x_mean
        y_normalized = y - y_mean

        norm = torch.sum(x_normalized * y_normalized, dim=(1, 2))
        var = torch.square(x_normalized).sum(dim=(1, 2)) * torch.square(y_normalized).sum(dim=(1, 2))
        denom = torch.sqrt(var + 1e-6)

        ncc = norm / (denom + 1e-6)

        # ignore pathces with low variances
        not_valid = (torch.square(x_normalized).sum(dim=(1, 2)) < self.min_patch_variance) | (
            torch.square(y_normalized).sum(dim=(1, 2)) < self.min_patch_variance
        )
        ncc[not_valid] = 1.0

        score = 1 - ncc.clip(-1.0, 1.0)  # 0->2: smaller, better
        return score[:, None, None, None]


class MultiViewLoss(nn.Module):
    """compute multi-view consistency loss"""

    def __init__(self, patch_size: int = 11, topk: int = 4, min_patch_variance: float = 0.01):
        super(MultiViewLoss, self).__init__()
        self.patch_size = patch_size
        self.topk = topk
        self.min_patch_variance = min_patch_variance
        # TODO make metric configurable
        # self.ssim = SSIM(patch_size=patch_size)
        # self.ncc = NCC(patch_size=patch_size)
        self.ssim = NCC(patch_size=patch_size, min_patch_variance=min_patch_variance)

        self.iter = 0

    def forward(self, patches: torch.Tensor, valid: torch.Tensor):
        """take the mim

        Args:
            patches (torch.Tensor): _description_
            valid (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        num_imgs, num_rays, _, num_channels = patches.shape

        if num_rays <= 0:
            return torch.tensor(0.0).to(patches.device)

        ref_patches = (
            patches[:1, ...]
            .reshape(1, num_rays, self.patch_size, self.patch_size, num_channels)
            .expand(num_imgs - 1, num_rays, self.patch_size, self.patch_size, num_channels)
            .reshape(-1, self.patch_size, self.patch_size, num_channels)
            .permute(0, 3, 1, 2)
        )  # [N_src*N_rays, 3, patch_size, patch_size]
        src_patches = (
            patches[1:, ...]
            .reshape(num_imgs - 1, num_rays, self.patch_size, self.patch_size, num_channels)
            .reshape(-1, self.patch_size, self.patch_size, num_channels)
            .permute(0, 3, 1, 2)
        )  # [N_src*N_rays, 3, patch_size, patch_size]

        # apply same reshape to the valid mask
        src_patches_valid = (
            valid[1:, ...]
            .reshape(num_imgs - 1, num_rays, self.patch_size, self.patch_size, 1)
            .reshape(-1, self.patch_size, self.patch_size, 1)
            .permute(0, 3, 1, 2)
        )  # [N_src*N_rays, 1, patch_size, patch_size]

        ssim = self.ssim(ref_patches.detach(), src_patches)
        ssim = torch.mean(ssim, dim=(1, 2, 3))
        ssim = ssim.reshape(num_imgs - 1, num_rays)

        # ignore invalid patch by setting ssim error to very large value
        ssim_valid = (
            src_patches_valid.reshape(-1, self.patch_size * self.patch_size).all(dim=-1).reshape(num_imgs - 1, num_rays)
        )
        # we should mask the error after we select the topk value, otherwise we might select far way patches that happens to be inside the image
        # ssim[torch.logical_not(ssim_valid)] = 1.1  # max ssim_error is 1

        min_ssim, idx = torch.topk(ssim, k=self.topk, largest=False, dim=0, sorted=True)

        min_ssim_valid = ssim_valid[idx, torch.arange(num_rays)[None].expand_as(idx)]
        # TODO how to set this value for better visualization
        min_ssim[torch.logical_not(min_ssim_valid)] = 0.0  # max ssim_error is 1

        if False:

            # visualization of topK error computations

            import cv2
            import numpy as np

            vis_patch_num = num_rays
            K = min(100, vis_patch_num)

            image = (
                patches[:, :vis_patch_num, :, :]
                .reshape(-1, vis_patch_num, self.patch_size, self.patch_size, 3)
                .permute(1, 2, 0, 3, 4)
                .reshape(vis_patch_num * self.patch_size, -1, 3)
            )

            src_patches_reshaped = src_patches.reshape(
                num_imgs - 1, num_rays, 3, self.patch_size, self.patch_size
            ).permute(1, 0, 3, 4, 2)
            idx = idx.permute(1, 0)

            selected_patch = (
                src_patches_reshaped[torch.arange(num_rays)[:, None].expand(idx.shape), idx]
                .permute(0, 2, 1, 3, 4)
                .reshape(num_rays, self.patch_size, self.topk * self.patch_size, 3)[:vis_patch_num]
                .reshape(-1, self.topk * self.patch_size, 3)
            )

            # apply same reshape to the valid mask
            src_patches_valid_reshaped = src_patches_valid.reshape(
                num_imgs - 1, num_rays, 1, self.patch_size, self.patch_size
            ).permute(1, 0, 3, 4, 2)

            selected_patch_valid = (
                src_patches_valid_reshaped[torch.arange(num_rays)[:, None].expand(idx.shape), idx]
                .permute(0, 2, 1, 3, 4)
                .reshape(num_rays, self.patch_size, self.topk * self.patch_size, 1)[:vis_patch_num]
                .reshape(-1, self.topk * self.patch_size, 1)
            )
            # valid to image
            selected_patch_valid = selected_patch_valid.expand_as(selected_patch).float()
            # breakpoint()

            image = torch.cat([selected_patch_valid, selected_patch, image], dim=1)
            # select top rays with highest errors

            image = image.reshape(num_rays, self.patch_size, -1, 3)

            _, idx2 = torch.topk(
                torch.sum(min_ssim, dim=0) / (min_ssim_valid.float().sum(dim=0) + 1e-6),
                k=K,
                largest=True,
                dim=0,
                sorted=True,
            )

            image = image[idx2].reshape(K * self.patch_size, -1, 3)

            cv2.imwrite(f"vis/{self.iter}.png", (image.detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
            self.iter += 1
            if self.iter == 9:
                breakpoint()

        return torch.sum(min_ssim) / (min_ssim_valid.float().sum() + 1e-6)


# sensor depth loss, adapted from https://github.com/dazinovic/neural-rgbd-surface-reconstruction/blob/main/losses.py
class SensorDepthLoss(nn.Module):
    """Sensor Depth loss"""

    def __init__(self, truncation: float):
        super(SensorDepthLoss, self).__init__()
        self.truncation = truncation  #  0.05 * 0.3 5cm scaled

    def forward(self, batch, outputs):
        """take the mim

        Args:
            batch (Dict): inputs
            outputs (Dict): outputs data from surface model

        Returns:
            l1_loss: l1 loss
            freespace_loss: free space loss
            sdf_loss: sdf loss
        """
        depth_pred = outputs["depth"]
        depth_gt = batch["sensor_depth"].to(depth_pred.device)[..., None]
        valid_gt_mask = depth_gt > 0.0

        l1_loss = torch.sum(valid_gt_mask * torch.abs(depth_gt - depth_pred)) / (valid_gt_mask.sum() + 1e-6)

        # free space loss and sdf loss
        ray_samples = outputs["ray_samples"]
        filed_outputs = outputs["field_outputs"]
        pred_sdf = filed_outputs[FieldHeadNames.SDF][..., 0]
        directions_norm = outputs["directions_norm"]

        z_vals = ray_samples.frustums.starts[..., 0] / directions_norm

        truncation = self.truncation
        front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        num_fs_samples = front_mask.sum()
        num_sdf_samples = sdf_mask.sum()
        num_samples = num_fs_samples + num_sdf_samples + 1e-6
        fs_weight = 1.0 - num_fs_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        free_space_loss = torch.mean((F.relu(truncation - pred_sdf) * front_mask) ** 2) * fs_weight

        sdf_loss = torch.mean(((z_vals + pred_sdf) - depth_gt) ** 2 * sdf_mask) * sdf_weight

        return l1_loss, free_space_loss, sdf_loss
