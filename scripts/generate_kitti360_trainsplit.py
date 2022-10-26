"""
generate kitti360 train split
"""

import os
from pathlib import Path
import numpy as np


train_split = [
    # seq, start, end
    (0, 347, 450),
    (0, 3540, 3665),
    (0, 3820, 3937),
    (0, 6190, 6290),
    (0, 7840, 7940),
    (2, 5950, 6050),
    (2, 7490, 7595),
    (2, 8065, 8165),
    (4, 135, 212),
    (4, 382, 482),
    (4, 1385, 1486),
    (4, 1741, 1843),
    (5, 1130, 1240),
    (5, 1928, 2035),
]

distance_interval_min = 0.8

out_dir = Path("tmp_data/kitti360_trainsplit/")
out_dir.mkdir(exist_ok=True, parents=True)

all_distances = []

for scene_id, (seq, first_frame, last_frame) in enumerate(train_split):
    pose_file = Path(
        f"/home/yuzh/mnt/mlcloud/Projects/download/KITTI-360/data_poses/2013_05_28_drive_{seq:04d}_sync/poses.txt"
    )
    poses = np.loadtxt(pose_file)
    pose_prev = None
    acc_dis = 0
    test_frame_dis_pre = None
    distances = [0]
    frames = []

    # go through all frames of this window
    for frame in range(first_frame, last_frame + 1):
        if not frame in list(poses[:, 0]):
            continue
        frames.append(frame)
        pose_i = poses[poses[:, 0] == frame, 1:].reshape(3, 4)
        if frame == first_frame:
            pose_prev = pose_i
        else:
            dis = np.linalg.norm(pose_i[:3, 3] - pose_prev[:3, 3])
            distances.append(dis)
            all_distances.append(dis)
            pose_prev = pose_i
            acc_dis += dis

    # if car drives faster than average speed
    distance_interval = max(distance_interval_min, np.mean(distances) - 0.5)

    buffer_distances = distance_interval * 10
    cum_distances = np.cumsum(distances)
    acc_dis_k = 0
    test_frames = []
    train_frames = []
    selected_frames = []
    selected_distances = []
    for k, frame in enumerate(frames):
        acc_dis_k += distances[k]
        # if acc_dis_k>10 and acc_dis_k<acc_dis-10:
        if True:
            if len(selected_frames) == 0:
                selected_frames.append(frame)
                selected_distances.append(acc_dis_k)
                test_frame_dis_pre = acc_dis_k
            else:
                if acc_dis_k - test_frame_dis_pre < distance_interval:
                    continue
                else:
                    selected_frames.append(frame)
                    selected_distances.append(acc_dis_k)
                    test_frame_dis_pre = acc_dis_k

    test_frames = [
        f
        for l, f in enumerate(selected_frames)
        if l % 2 == 1 and selected_distances[l] > 20 and selected_distances[l] < acc_dis - 20
    ]
    train_frames = [f for l, f in enumerate(selected_frames) if l % 2 == 0]

    print("Train:", train_frames)
    print("Test:", test_frames)

    train_image_dir = out_dir / Path(f"train_{scene_id:02d}")
    train_image_dir.mkdir(exist_ok=True)

    test_image_dir = out_dir / Path(f"test_{scene_id:02d}")
    test_image_dir.mkdir(exist_ok=True)

    train_list_file = out_dir / Path(f"train_{scene_id:02d}.txt")
    test_list_file = out_dir / Path(f"test_{scene_id:02d}.txt")

    for image_dir, list_file, frames in [
        (train_image_dir, train_list_file, train_frames),
        (test_image_dir, test_list_file, test_frames),
    ]:
        with open(list_file, "w", encoding="utf8") as f:
            for frame in frames:
                f.write(f"2013_05_28_drive_{seq:04d}_sync/image_00/data_rect/{frame:010d}.png\n")
                image_dir_i = image_dir / f"2013_05_28_drive_{seq:04d}_sync" / "image_00" / "data_rect"
                image_dir_i.mkdir(exist_ok=True, parents=True)
                cmd = f"cp /home/yuzh/mnt/mlcloud/Projects/download/KITTI-360/KITTI-360/data_2d_raw/2013_05_28_drive_{seq:04d}_sync/image_00/data_rect/{frame:010d}.png {image_dir_i}"
                print(cmd)
                os.system(cmd)

                image_dir_i = image_dir / f"2013_05_28_drive_{seq:04d}_sync" / "image_01" / "data_rect"
                image_dir_i.mkdir(exist_ok=True, parents=True)
                cmd = f"cp /home/yuzh/mnt/mlcloud/Projects/download/KITTI-360/KITTI-360/data_2d_raw/2013_05_28_drive_{seq:04d}_sync/image_01/data_rect/{frame:010d}.png {image_dir_i}"
                print(cmd)
                os.system(cmd)

print("Average distance %f" % np.mean(all_distances))
