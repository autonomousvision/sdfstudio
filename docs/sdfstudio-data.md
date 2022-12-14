# Data

This is a short documentation of sdfstudio data and it is organized as:

- [Dataset format](#Dataset-format)
- [Existing datasets](#Existing-dataset)
- [Customize your own dataset](#Custom-dataset)

# Dataset format
We use scan65 of DTU scene to show how the data are organized. It looks like the following:
```bash
└── scan65
  └── meta_data.json
  ├── pairs.txt
  ├── 000000_rgb.png
  ├── 000000_normal.npy
  ├── 000000_depth.npy
  ├── .....
```
The json file (meta_data.json) stores meta data of the scene, it has the following format:
```yaml
{
    "camera_model": "OPENCV",   # camera model, currently only opencv is supported
    "height": 384,              # height of the images
    "width": 384,               # width of the images
    "has_mono_prior": true,     # contains monocualr prior or not
    "pairs": "paris.txt",       # pairs file used for multi-view photometric consistency loss
    "worldtogt": [[ 1, 0, 0, 0], # world to gt transformation, it's usefule for evauation
                  [ 0, 1, 0, 0],
                  [ 0, 0, 1, 0],
                  [ 0, 0, 0, 1]],
    "scene_box": {
        "aabb": [[-1, -1, -1],  # aabb for the bbox
                 [1, 1, 1]],
        "near": 0.5,            # near plane for each image
        "far": 4.5,             # far plane for each image
        "radius": 1.0,          # radius of ROI region in scene
        "collider_type": "near_far"   
        # collider_type can be "near_far", "box", "sphere", 
        # it indicates how do we determine the near and far for each ray 
        # 1. near_far means we use the same near and far value for each ray
        # 2. box means we compute the intersection with bbox 
        # 3. sphere means we compute the intersection with sphere
    },
    "frames": [   # this contains information for each image
        {
            # note all paths are relateive path
            # path of rgb image
            "rgb_path": "000000_rgb.png",   
            # camera to world transform
            "camtoworld": [[0.9702627062797546, -0.014742869883775711, -0.2416049987077713, 0.6601868867874146],
                           [0.007479910273104906, 0.9994929432868958, -0.03095100075006485, 0.07803472131490707],
                           [0.2419387847185135, 0.028223417699337006, 0.9698809385299683, -2.6397712230682373],
                           [0.0, 0.0, 0.0, 1.0 ]],
            # intrinsic of current imaga
            "intrinsics": [[925.5457763671875, -7.8512319305446e-05, 199.4256591796875, 0.0],
                           [0.0, 922.6160278320312, 198.10269165039062, 0.0 ],
                           [0.0, 0.0, 1.0, 0.0 ],
                           [0.0, 0.0, 0.0, 1.0 ]],
            # path of monocular depth prior
            "mono_depth_path": "000000_depth.npy",
            # path of monocular normal prior
            "mono_normal_path": "000000_normal.npy"
        },
        ...
    ]    
}
```

The `paris.txt` is used for multi-view photometric consistency loss. It has the following format:
```bash
# ref image, source image 1, source image 2, ..., source image N
000000.png 000032.png 000023.png 000028.png 000031.png 000029.png 000030.png 000024.png 000002.png 000015.png 000025.png ...
000001.png 000033.png 000003.png 000022.png 000016.png 000027.png 000023.png 000007.png 000011.png 000026.png 000024.png ...
...
```
# Existing datasets

We adapted the dataset used in MonoSDF to sdfstudio format and it can be downloaded with
```
ns-download-data sdfstudio --dataset-name DATASET_NAME
```
The `DATASET_NAME` can be chosen from `sdfstudio-demo-data, dtu, replica, scannet, tanks-and-temple, tanks-and-temple-highres, all`. Use all if you want to download all the dataset.

Note that for the DTU dataset, you should use `--pipeline.model.sdf-field.inside-outside False` and for the indoor dataset you should use `--pipeline.model.sdf-field.inside-outside True` druing training.

We also provide the preprocessed heritage data from neuralreconW and it can be downloaded with
```
ns-download-data sdfstudio --dataset-name heritage
```

# Customize your own dataset

You could implement your own data-parser to use custom dataset or convert you dataset to sdfstudio data format as shown above. Here we provide an example to convert scannet dataset to sdfstudio data format. Please change the path accordingly.
```bash
python scripts/datasets/process_scannet_to_sdfstudio.py --input_path /home/yuzh/Projects/datasets/scannet/scene0050_00 --output_path data/custom/scannet_scene0050_00
```

Next, you can extract monocular depths and normals (please install [omnidata model](https://github.com/EPFL-VILAB/omnidata) before running the command):
```bash
python scripts/datasets/extract_monocular_cues.py --task normal --img_path data/custom/scannet_scene0050_00/ --output_path data/custom/scannet_scene0050_00 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
python scripts/datasets/extract_monocular_cues.py --task normal --img_path data/custom/scannet_scene0050_00/ --output_path data/custom/scannet_scene0050_00 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
```