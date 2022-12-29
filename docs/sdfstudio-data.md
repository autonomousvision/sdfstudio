# Data Format and Datasets

This is a short documentation of SDFStudio's data format and datasets, organized as follows:

- [Data format](#Dataset-format)
- [Existing datasets](#Existing-dataset)
- [Customize your own dataset](#Customize-your-own-dataset)

# Data Format

We use scan65 of the DTU dataset to show how SDF Studio's data structures are organized:

```bash
└── scan65
  └── meta_data.json
  ├── pairs.txt
  ├── 000000_rgb.png
  ├── 000000_normal.npy
  ├── 000000_depth.npy
  ├── .....
```

The json file (meta_data.json) stores meta data of the scene and has the following format:

```yaml
{
  'camera_model': 'OPENCV', # camera model (currently only OpenCV is supported)
  'height': 384, # height of the images
  'width': 384, # width of the images
  'has_mono_prior': true, # use monocular cues or not
  'pairs': 'pairs.txt', # pairs file used for multi-view photometric consistency loss
  'worldtogt': [
      [1, 0, 0, 0], # world to gt transformation (useful for evauation)
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
  'scene_box': {
      'aabb': [
          [-1, -1, -1], # aabb for the bbox
          [1, 1, 1],
        ],
      'near': 0.5, # near plane for each image
      'far': 4.5, # far plane for each image
      'radius': 1.0, # radius of ROI region in scene
      'collider_type': 'near_far',
      # collider_type can be "near_far", "box", "sphere",
      # it indicates how do we determine the near and far for each ray
      # 1. near_far means we use the same near and far value for each ray
      # 2. box means we compute the intersection with the bounding box
      # 3. sphere means we compute the intersection with the sphere
    },
  'frames': [ # this contains information for each image
      {
        # note that all paths are relateive path
        # path of rgb image
        'rgb_path': '000000_rgb.png',
        # camera to world transform
        'camtoworld':
          [
            [
              0.9702627062797546,
              -0.014742869883775711,
              -0.2416049987077713,
              0.6601868867874146,
            ],
            [
              0.007479910273104906,
              0.9994929432868958,
              -0.03095100075006485,
              0.07803472131490707,
            ],
            [
              0.2419387847185135,
              0.028223417699337006,
              0.9698809385299683,
              -2.6397712230682373,
            ],
            [0.0, 0.0, 0.0, 1.0],
          ],
        # intrinsic of current image
        'intrinsics':
          [
            [925.5457763671875, -7.8512319305446e-05, 199.4256591796875, 0.0],
            [0.0, 922.6160278320312, 198.10269165039062, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
          ],
        # path of monocular depth prior
        'mono_depth_path': '000000_depth.npy',
        # path of monocular normal prior
        'mono_normal_path': '000000_normal.npy',
      },
      ...,
    ],
}
```

The file `pairs.txt` is used for the multi-view photometric consistency loss and has the following format:

```bash
# ref image, source image 1, source image 2, ..., source image N, note source image are listed in ascending order, which means last image has largest score
000000.png 000032.png 000023.png 000028.png 000031.png 000029.png 000030.png 000024.png 000002.png 000015.png 000025.png ...
000001.png 000033.png 000003.png 000022.png 000016.png 000027.png 000023.png 000007.png 000011.png 000026.png 000024.png ...
...
```

# Existing datasets

We adapted the datasets used in MonoSDF to the SDFStudio format. They can be downloaded as follows:

```
ns-download-data sdfstudio --dataset-name DATASET_NAME
```

Here, `DATASET_NAME` can be any of the following: `sdfstudio-demo-data, dtu, replica, scannet, tanks-and-temple, tanks-and-temple-highres, all`. Use `all` if you want to download all datasets.

Note that for the DTU dataset, you should use `--pipeline.model.sdf-field.inside-outside False` and for the indoor datasets (Replica, ScanNet, Tanks and Temples) you should use `--pipeline.model.sdf-field.inside-outside True` during training.

We also provide the preprocessed heritage data from NeuralReconW which can be downloaded as follows:

```
ns-download-data sdfstudio --dataset-name heritage
```

## RGBD data

SDFStudio also supports RGB-D data to obtain high-quality 3D reconstruction. The [synthetic rgbd data](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) can be downloaded as follows

```
ns-download-data sdfstudio --dataset-name neural-rgbd-data
```

Then run the following command to convert the downloaded neural-rgbd dataset to SDFStudio format:

```bash
# kitchen scene for example, replca the scene path to convert other scenes
python scripts/datasets/process_neuralrgbd_to_sdfstudio.py --input_path data/neural-rgbd-data/kitchen/ --output_path data/neural_rgbd/kitchen_sensor_depth --type sensor_depth
```

# Customize your own dataset

You can implement your own data parser to use your own dataset or convert your dataset to SDFStudio's data format. Here, we provide an example for converting the ScanNet dataset to SDF Studio's data format.

```bash
python scripts/datasets/process_scannet_to_sdfstudio.py --input_path /your_path/datasets/scannet/scene0050_00 --output_path data/custom/scannet_scene0050_00
```

Next, you can extract monocular depth and normal cues if you like to use those during optimization. First, install the [Omnidata Model](https://github.com/EPFL-VILAB/omnidata). Next, run the following command:

```bash
python scripts/datasets/extract_monocular_cues.py --task normal --img_path data/custom/scannet_scene0050_00/ --output_path data/custom/scannet_scene0050_00 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
python scripts/datasets/extract_monocular_cues.py --task depth --img_path data/custom/scannet_scene0050_00/ --output_path data/custom/scannet_scene0050_00 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
```
