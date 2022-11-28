#!/usr/bin/env python
"""Convert ETH3D to NerfStudio data format"""

import json
import os
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pyrender
import torch
import trimesh
import yaml
from tqdm import tqdm

# from nerfstudio.utils import colmap_utils
from nerfstudio.data.utils import colmap_utils
from nerfstudio.model_components.ray_samplers import save_points

os.environ["PYOPENGL_PLATFORM"] = "egl"


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
}

palette = np.random.randint(0, 255, size=(200, 3))
# palette = np.arange(0, 200).reshape(-1, 1)  # .expand((200, 3))

palette = np.array(palette)


def show_result(seg):

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]
    return color_seg


class Renderer:
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(
            cx=intrinsics[0, 2], cy=intrinsics[1, 2], fx=intrinsics[0, 0], fy=intrinsics[1, 1]
        )
        self.scene.add(cam, pose=self.fix_pose(pose))
        # flags = pyrender.constants.RenderFlags.OFFSCREEN
        return self.renderer.render(self.scene, flags=self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


def colmap_to_json(
    scene_path: Path,
    sfm: Path,
    camera_model: CameraModel,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        cameras_path: Path to the cameras.bin file.
        images_path: Path to the images.bin file.
        output_dir: Path to the output directory.
        camera_model: Camera model used.

    Returns:
        The number of registered images.
    """
    cameras_path = scene_path / sfm / "cameras.bin"
    images_path = scene_path / sfm / "images.bin"
    points_path = scene_path / sfm / "points3D.bin"
    config_path = scene_path / "config.yaml"

    with open(config_path, "r") as yamlfile:
        scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    radius = scene_config["radius"]
    origin = np.array(scene_config["origin"]).reshape(1, 3)

    cams = colmap_utils.read_cameras_binary(cameras_path)
    imgs = colmap_utils.read_images_binary(images_path)
    pts3d = colmap_utils.read_points3d_binary(points_path)

    # pts3d_array = np.array([pts3d[p_id].xyz for p_id in pts3d])
    # error_array = np.array([pts3d[p_id].error for p_id in pts3d])

    # key point depth
    pts3d_array = torch.ones(max(pts3d.keys()) + 1, 4)
    error_array = torch.ones(max(pts3d.keys()) + 1, 1)
    for pts_id, pts in tqdm(pts3d.items()):
        pts3d_array[pts_id, :3] = torch.from_numpy(pts.xyz)
        error_array[pts_id, 0] = torch.from_numpy(pts.error)

    points_ori = []
    min_track_length = scene_config["min_track_length"]
    for id, p in pts3d.items():
        if p.point2D_idxs.shape[0] > min_track_length:
            points_ori.append(p.xyz)
    points_ori = np.array(points_ori)
    save_points("nori_3.ply", points_ori)

    points_ori -= origin
    print(points_ori.shape)

    # expand and quantify
    points_ori = torch.from_numpy(points_ori)
    offset = torch.linspace(-1, 1.0, 3)
    offset_cube = torch.meshgrid(offset, offset, offset)
    offset_cube = torch.stack(offset_cube, dim=-1).reshape(-1, 3)

    voxel_size = scene_config["voxel_size"]
    offset_cube *= voxel_size  # voxel size
    expand_points = points_ori[:, None, :] + offset_cube[None]
    expand_points = expand_points.reshape(-1, 3)
    save_points("expand_points.ply", expand_points.numpy())

    # filter
    # filter out points out of [-1, 1]
    mask = torch.prod((expand_points > -radius), axis=-1, dtype=torch.bool) & torch.prod(
        (expand_points < radius), axis=-1, dtype=torch.bool
    )
    filtered_points = expand_points[mask]
    save_points("filtered_points.ply", filtered_points.numpy())

    grid_size = 32
    voxel_size = 2 * radius / grid_size
    quantified_points = torch.floor(((filtered_points / radius) + 1.0) * grid_size // 2)

    index = quantified_points[:, 0] + quantified_points[:, 1] * grid_size + quantified_points[:, 2] * grid_size**2

    offset = torch.linspace(-radius + voxel_size / 2.0, radius - voxel_size / 2.0, grid_size)
    z, y, x = torch.meshgrid(offset, offset, offset, indexing="xy")
    offset_cube = torch.stack([x, z, y], dim=-1).reshape(-1, 3)

    mask = torch.zeros(grid_size**3, dtype=torch.bool)
    mask[index.long()] = True

    points_valid = offset_cube[mask]
    save_points("quantified_points.ply", points_valid.numpy())
    # breakpoint()

    """
    xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
    xyz_world_error = np.array([pts3d[p_id].error for p_id in pts3d])
    xyz_world = xyz_world[xyz_world_error < 0.2]
    sfm2gt = np.array(scene_config["sfm2gt"])
    xyz_world = xyz_world @ sfm2gt[:3, :3].T + sfm2gt[:3, 3:].T
    save_points("pp.ply", xyz_world)
    """

    mesh = trimesh.creation.icosphere(5, radius=radius)
    mesh.vertices = mesh.vertices + np.array(scene_config["origin"]).reshape(1, 3)

    meshes = []
    for p in points_valid:
        box = trimesh.creation.box(extents=(voxel_size, voxel_size, voxel_size))
        box.vertices = box.vertices + origin + p.numpy().reshape(-1, 3)
        meshes.append(box)

    mesh = trimesh.util.concatenate(meshes)
    mesh.export("box.ply")

    """
    vertices = mesh.vertices @ sfm2gt[:3, :3].T + sfm2gt[:3, 3:].T
    save_points("sphere.ply", vertices)
    """

    # print(cameras)
    poses = []
    fxs = []
    fys = []
    cxs = []
    cys = []
    image_filenames = []
    mask_filenames = []
    masks = []

    data = scene_path

    for _id, cam in cams.items():
        img = imgs[_id]

        assert cam.model == "PINHOLE", "Only pinhole (perspective) camera model is supported at the moment"

        pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
        pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
        poses.append(torch.linalg.inv(pose))
        fxs.append(torch.tensor(cam.params[0]))
        fys.append(torch.tensor(cam.params[1]))
        cxs.append(torch.tensor(cam.params[2]))
        cys.append(torch.tensor(cam.params[3]))

        image_filenames.append(data / "dense/images" / img.name)
        mask_filenames.append(data / "semantic_maps" / img.name.replace(".jpg", ".npz"))

        # visualize pts3d for each image
        valid_3d_mask = img.point3D_ids != -1
        point3d_ids = img.point3D_ids[valid_3d_mask]
        img_p3d = pts3d_array[point3d_ids]
        img_err = error_array[point3d_ids]

        # img_p3d = img_p3d[img_err[:, 0] < torch.median(img_err)]
        save_points(f"W/{_id}_nof.ply", img_p3d.cpu().numpy()[:, :3])

        # render bounding sphere mask
        renderer = Renderer()
        mesh_opengl = renderer.mesh_opengl(mesh)

        intrinsic = np.eye(4)
        intrinsic[0, 0] = cam.params[0]
        intrinsic[1, 1] = cam.params[1]
        intrinsic[0, 2] = cam.params[2]
        intrinsic[1, 2] = cam.params[3]

        H = cam.height
        W = cam.width
        pose = poses[-1].cpu().numpy()

        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
        print(intrinsic)
        print(pose)
        print(depth_pred.min(), depth_pred.max())
        renderer.delete()

        mask = np.load(mask_filenames[-1])["arr_0"]
        semantic_image = show_result(mask)

        # ['person', 'car', 'bicycle', 'minibike'] with id [12, 20,127,116]
        # ['sky'] = 2
        # new mask [80, 83, 43, 41, 115, 110]
        semantic_ids_to_skip = [12, 20, 127, 116]  # + [80, 83, 43, 41, 115, 110]  # + [2]
        mask = np.stack([mask != semantic_id for semantic_id in semantic_ids_to_skip])  # + mask2

        mask = mask.all(axis=0)

        rgb_img = cv2.imread(str(image_filenames[-1]))
        print(rgb_img.shape, mask.shape, H, W)
        if rgb_img.shape[0] != H and rgb_img.shape[1] != W:
            print("warning")
            continue
        rgb_img_masked_semantic = rgb_img * mask[..., None]

        depth_mask = depth_pred > 0.0001
        rgb_img_masked = rgb_img * depth_mask[..., None]

        mask = depth_mask & mask

        rgb_img = rgb_img * mask[..., None]
        image = np.concatenate((rgb_img, rgb_img_masked_semantic, semantic_image, rgb_img_masked), axis=1)

        # cv2.imshow("ssdf", image)
        # cv2.waitKey(0)

        # write mask
        (scene_path / "masks").mkdir(exist_ok=True, parents=False)
        np.save(scene_path / "masks" / img.name.replace(".jpg", ".npy"), mask)


scene_path = Path("data/Heritage-Recon/brandenburg_gate/")
scene_path = Path("data/Heritage-Recon/lincoln_memorial/")
scene_path = Path("data/Heritage-Recon/pantheon_exterior/")
# scene_path = Path("data/Heritage-Recon/palacio_de_bellas_artes/")

# sfm = "neuralsfm"
sfm = "dense/sparse"

colmap_to_json(
    scene_path=scene_path,
    sfm=sfm,
    camera_model=CameraModel.OPENCV,
)
