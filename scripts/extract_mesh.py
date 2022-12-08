#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.marching_cubes import get_surface_sliding

CONSOLE = Console(width=120)


@dataclass
class ExtractMesh:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Marching cube resolution.
    resolution: int = 1024
    # Name of the output file.
    output_path: Path = Path("output.ply")
    # Whether to simplify the mesh.
    simplify_mesh: bool = True

    def main(self) -> None:
        """Main function."""
        assert self.resolution % 512 == 0

        _, pipeline, _ = eval_setup(self.load_config)

        get_surface_sliding(
            sdf=lambda x: pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous(),
            resolution=self.resolution,
            grid_boundary=[-1.0, 1.0],
            coarse_mask=pipeline.model.scene_box.coarse_binary_gird,
            output_path=self.output_path,
            simplify_mesh=self.simplify_mesh,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[ExtractMesh]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ExtractMesh)  # noqa
