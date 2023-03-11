#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Name of the output images dir.
    output_images_path: Path = Path("output_images/")

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        metrics_dict, images_dict_list = pipeline.get_average_eval_image_metrics()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_images_path.mkdir(parents=True, exist_ok=True)

        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")

        for idx, images_dict in enumerate(images_dict_list):
            for k, v in images_dict.items():
                cv2.imwrite(
                    str(self.output_images_path / Path(f"{k}_{idx}.png")),
                    (v.cpu().numpy() * 255.0).astype(np.uint8)[..., ::-1],
                )
        CONSOLE.print(f"Saved rendering results to: {self.output_images_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
