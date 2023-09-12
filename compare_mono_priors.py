import argparse
from pathlib import Path

import cv2
import rerun as rr


def main(lowres_path:Path, highres_path:Path):
    lowres_img_paths = sorted(lowres_path.glob("*_rgb.png"))
    highres_img_paths = sorted(highres_path.glob("*_rgb.png"))
    lowres_depth_paths = sorted(lowres_path.glob("*_depth.png"))
    highres_depth_paths = sorted(highres_path.glob("*_depth.png"))
    lowres_normal_paths = sorted(lowres_path.glob("*_normal.png"))
    highres_normal_paths = sorted(highres_path.glob("*_normal.png"))

    num_samples = len(lowres_img_paths)

    for i in range(num_samples):
        rr.set_time_sequence("idx", i)
        lowres_img = cv2.imread(str(lowres_img_paths[i]))
        highres_img = cv2.imread(str(highres_img_paths[i]))
        lowres_depth = cv2.imread(str(lowres_depth_paths[i]))
        highres_depth = cv2.imread(str(highres_depth_paths[i]))
        lowres_normal = cv2.imread(str(lowres_normal_paths[i]))
        highres_normal = cv2.imread(str(highres_normal_paths[i]))

        # log priors togther to visualize easily
        rr.log_image("compare-rgb/lowres", lowres_img[..., ::-1])
        rr.log_image("compare-rgb/highres", highres_img[..., ::-1])
        # rr.log_image("compare-depth/lowres", lowres_depth[..., ::-1])
        # rr.log_image("compare-depth/highres", highres_depth[..., ::-1])
        # rr.log_image("compare-normal/lowres", lowres_normal[..., ::-1])
        # rr.log_image("compare-normal/highres", highres_normal[..., ::-1])

        rr.log_image("depth-lowres", lowres_depth[..., ::-1])
        rr.log_image("depth-highres", highres_depth[..., ::-1])
        rr.log_image("normal-lowres", lowres_normal[..., ::-1])
        rr.log_image("normal-highres", highres_normal[..., ::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare mono priors")
    parser.add_argument('--lowres-path', type=Path, required=True, help='Path to directory containing lowres data')
    parser.add_argument('--highres-path', type=Path, required=True, help='Path to directory containing highres data')
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "my_application")
    main(args.lowres_path, args.highres_path)
    rr.script_teardown(args)

