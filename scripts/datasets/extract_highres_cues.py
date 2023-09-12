import torch
import numpy as np
import cv2
import os
import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# copy from vis-mvsnet
def find_files(dir, exts=['*.png', '*.jpg']):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

# copy from vis-mvsnet
def load_cam(file: str):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    with open(file) as f:
        words = f.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    return cam

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
            
# adatpted from https://github.com/dakshaau/ICP/blob/master/icp.py#L4 for rotation only 
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    AA = A
    BB = B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    return R


#TODO merge the following 4 function to one single function

# align depth map in the x direction from left to right
def align_x(depth1, depth2, s1, e1, s2, e2):
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[1] == depth2.shape[1]

    assert (e1 - s1) == (e2 - s2), f"{e1 - s1} | {e2-s2}"
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, :, s2:e2], depth1[:, :, s1:e1], torch.ones_like(depth1[:, :, s1:e1]))

    depth2_aligned = scale * depth2 + shift   
    result = torch.ones((1, depth1.shape[1], depth1.shape[2] + depth2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = depth1[:, :, :s1]
    result[:, :, depth1.shape[2]:] = depth2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1-s1))[None, None, :]
    result[:, :, s1:depth1.shape[2]] = depth1[:, :, s1:] * weight + depth2_aligned[:, :, :e2] * (1 - weight)

    return result

# align depth map in the y direction from top to down
def align_y(depth1, depth2, s1, e1, s2, e2):
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[2] == depth2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, s2:e2, :], depth1[:, s1:e1, :], torch.ones_like(depth1[:, s1:e1, :]))

    depth2_aligned = scale * depth2 + shift   
    result = torch.ones((1, depth1.shape[1] + depth2.shape[1] - (e1 - s1), depth1.shape[2]))

    result[:, :s1, :] = depth1[:, :s1, :]
    result[:, depth1.shape[1]:, :] = depth2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1-s1))[None, :, None]
    result[:, s1:depth1.shape[1], :] = depth1[:, s1:, :] * weight + depth2_aligned[:, :e2, :] * (1 - weight)

    return result

# align normal map in the x direction from left to right
def align_normal_x(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[1] == normal2.shape[1]

    assert (e1 - s1) == (e2 - s2)
    
    R = best_fit_transform(normal2[:, :, s2:e2].reshape(3, -1).T, normal1[:, :, s1:e1].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1], normal1.shape[2] + normal2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = normal1[:, :, :s1]
    result[:, :, normal1.shape[2]:] = normal2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1-s1))[None, None, :]
    
    result[:, :, s1:normal1.shape[2]] = normal1[:, :, s1:] * weight + normal2_aligned[:, :, :e2] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    
    return result

# align normal map in the y direction from top to down
def align_normal_y(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[2] == normal2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    
    R = best_fit_transform(normal2[:, s2:e2, :].reshape(3, -1).T, normal1[:, s1:e1, :].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1] + normal2.shape[1] - (e1 - s1), normal1.shape[2]))

    result[:, :s1, :] = normal1[:, :s1, :]
    result[:, normal1.shape[1]:, :] = normal2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1-s1))[None, :, None]
    
    result[:, s1:normal1.shape[1], :] = normal1[:, s1:, :] * weight + normal2_aligned[:, :e2, :] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    
    return result

def create_patches(image_path:Path, image_dir:Path, out_index: int)-> None:
    image = cv2.imread(str(image_path))
    # assume square?
    H, W = image.shape[:2]
    
    assert H == W == 384*2, f"image size is not 384*2, but {H}x{W}"
    size = 384
    overlap = 128 # (128 + 128) -> 256 of overlap (384/3) each side overlapped with a middle section untouched
    x = W // overlap
    y = H // overlap


    # crop images
    for j in range(y-2):
        for i in range(x-2):
            image_cur = image[j*overlap:j*overlap+size, i*overlap:i*overlap+size, :]
            # add _rgb to the end of the file name so that extract_monocular_cues.py can find it
            target_file = image_dir / "patches" / f"{out_index:06d}_{j:02d}_{i:02d}_rgb.png"
            target_file.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(target_file), image_cur)


def merge_patches(image_dir, out_index, depth_patches_path, normal_patches_path):

    H, W = 768, 768
    
    assert H == W == 384*2
    overlap = 128 # (128 + 128) -> 256 of overlap (384/3) each side overlapped with a middle section untouched
    x = W // overlap
    y = H // overlap

    # align depth map
    depths_row = []
    # align depth maps from left to right row by row
    for j in range(y-2):            
        depths = []
        for i in range(x-2):
            # depth_path = os.path.join(out_path, "%06d_%02d_%02d_depth.npy"%(out_index, j, i))
            depth_path = depth_patches_path / f"{out_index:06d}_{j:02d}_{i:02d}_depth.npy"
            depth = np.load(depth_path)
            depth = torch.from_numpy(depth)[None]
            depths.append(depth)
        
        # align from left to right
        depth_left = depths[0]
        s1 = 128
        s2 = 0
        e2 = 128 * 2
        for depth_right in depths[1:]:
            depth_left = align_x(depth_left, depth_right, s1, depth_left.shape[2], s2, e2)
            s1 += 128
        depths_row.append(depth_left)

    depth_top = depths_row[0]
    # align depth maps from top to down
    s1 = 128
    s2 = 0
    e2 = 128 * 2
    for depth_bottom in depths_row[1:]:
        depth_top = align_y(depth_top, depth_bottom, s1, depth_top.shape[1], s2, e2)
        s1 += 128

    depth_top = (depth_top - depth_top.min()) / (depth_top.max() - depth_top.min())

    # final_depth_path = os.path.join(out_path_for_training ,"%06d_depth.png"%(out_index))
    final_depth_path = image_dir #/ "final_depth"

    final_depth_path.mkdir(exist_ok=True)
    plt.imsave(final_depth_path / f"{out_index:06d}_depth.png", depth_top[0].numpy(), cmap='viridis')
    np.save(final_depth_path / f"{out_index:06d}_depth.npy", depth_top.detach().cpu().numpy()[0])


    # normal
    normals_row = []
    # align normal maps from left to right row by row  
    for j in range(y-2):            
        normals = []
        for i in range(x-2):
            # normal_path = os.path.join(out_path, "%06d_%02d_%02d_normal.npy"%(out_index, j, i))
            normal_path = normal_patches_path / f"{out_index:06d}_{j:02d}_{i:02d}_normal.npy"
            normal = np.load(normal_path)
            normal = normal * 2. - 1.
            normal = normal / (np.linalg.norm(normal, axis=0) + 1e-15)[None]
            normals.append(normal)
        
        # align from left to right
        normal_left = normals[0]
        s1 = 128
        s2 = 0
        e2 = 128 * 2
        for normal_right in normals[1:]:
            normal_left = align_normal_x(normal_left, normal_right, s1, normal_left.shape[2], s2, e2)
            s1 += 128
        normals_row.append(normal_left)

    normal_top = normals_row[0]
    # align normal maps from top to down
    s1 = 128
    s2 = 0
    e2 = 128 * 2
    for normal_bottom in normals_row[1:]:
        normal_top = align_normal_y(normal_top, normal_bottom, s1, normal_top.shape[1], s2, e2)
        s1 += 128


    final_normal_path = image_dir #/ "final_normal"
    final_normal_path.mkdir(exist_ok=True)
    plt.imsave(final_normal_path / f"{out_index:06d}_normal.png", np.moveaxis(normal_top, [0,1, 2], [2, 0, 1]) * 0.5 + 0.5)
    np.save(final_normal_path / f"{out_index:06d}_normal.npy", (normal_top + 1.) / 2.)

def generate_monocular_priors(image_dir:Path,
                              omnidata_path, 
                              pretrained_models):
    assert image_dir.exists()

    image_paths = sorted(image_dir.glob("*_rgb.png"))
    patches_path = image_dir / "patches"

    out_index = 0
    # create_patches
    if not patches_path.exists():
        for image_path in tqdm(image_paths, desc="Creating Patches"):
            create_patches(image_path, image_dir, out_index)
            out_index += 1

    # merge patches
    out_index = 0
    # predict normals/depths for all patches
    depth_patches_path = image_dir / 'depth_patches'
    normal_patches_path = image_dir / 'normal_patches'

    num_patches = len(list(patches_path.glob("*.png")))
    num_depth_patches = len(list(depth_patches_path.glob('*.png')))
    num_normal_patches = len(list(normal_patches_path.glob('*.png')))

    # check to make sure that monocular patches don't already exist
    if num_patches != num_depth_patches:

        print("Generating mono depth...")
        os.system(
            f"python scripts/datasets/extract_monocular_cues.py \
            --omnidata_path {omnidata_path} \
            --pretrained_model {pretrained_models} \
            --img_path {image_dir / 'patches'} --output_path {depth_patches_path} \
            --task depth"
        )
    if num_patches != num_normal_patches:

        print("Generating mono normal...")
        os.system(
            f"python scripts/datasets/extract_monocular_cues.py \
            --omnidata_path {omnidata_path} \
            --pretrained_model {pretrained_models} \
            --img_path {image_dir / 'patches'} --output_path {normal_patches_path} \
            --task normal"
        )
    for image_path in tqdm(image_paths, desc="Merging Patches"):
        merge_patches(image_dir, out_index, depth_patches_path, normal_patches_path)
        out_index += 1

def main():
    parser = argparse.ArgumentParser(description='Generate high resolution outputs')

    parser.add_argument("--image-dir", required=True, help="directory containing images", type=Path)
    parser.add_argument("--omnidata-path", dest="omnidata_path",
                        default="/home/user/omnidata/omnidata_tools/torch/",
                        help="path to omnidata model")
    parser.add_argument("--pretrained-models", dest="pretrained_models",
                        default="/home/user/omnidata//omnidata_tools/torch/pretrained_models/",
                        help="path to pretrained models")
    args = parser.parse_args()

    generate_monocular_priors(args.image_dir,args.omnidata_path, args.pretrained_models)


if __name__ == "__main__":
    main()