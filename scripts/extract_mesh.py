#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import torch
import tyro
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.marching_cubes import get_surface_sliding, get_surface_occupancy

CONSOLE = Console(width=120)


@dataclass
class ExtractMesh:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"

        # breakpoint()
        get_surface_occupancy(occupancy_fn=lambda x: -pipeline.model.occupancy_field.forward_geonetwork(x)[:, 0])
        # get_surface_sliding(sdf=lambda x: -pipeline.model.occupancy_field.forward_geonetwork(x)[:, 0])

        # resolution=resolution,
        # grid_boundary=grid_boundary,
        # level=level

        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ExtractMesh).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ExtractMesh)  # noqa
