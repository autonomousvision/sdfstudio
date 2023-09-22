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

"""Utility functions to allow easy re-use of common operations across dataloaders"""
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image


def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype="int64").view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def create_masked_img(img_filepath: Path, mask_filepath: Path, output_dir: Path) -> Path:
    """
    Utility function to mask an image using provided mask and store it on disk.
    Output_dir is absolute path where to store the masked image.
    """
    img = np.array(Image.open(img_filepath), dtype=np.float32)
    mask = np.array(Image.open(mask_filepath), dtype=np.float32) / 255.0
    assert len(img.shape) == 3
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # in case the mask comes with alpha channel
    if mask.shape[-1] == 4:
        mask = mask[:, :, :3]

    if len(mask.shape) == 2:
        mask = mask[..., np.newaxis]

    masked_image = Image.fromarray((img * mask).astype(np.uint8))
    masked_image_filename = output_dir / (img_filepath.stem + "_masked" + img_filepath.suffix)
    masked_image.save(masked_image_filename)

    return masked_image_filename
