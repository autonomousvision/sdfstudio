# Documentation

This is a short documentation of SDFStudio, organized as follows:

- [Methods](#Methods)
- [Representations](#Representations)
- [Supervision](#Supervision)

# Methods

SDF Studio implements multiple neural implicit surface reconstruction methods in one common framework. More specifically, SDF Studio builds on [UniSurf](https://github.com/autonomousvision/unisurf), [VolSDF](https://github.com/lioryariv/volsdf), and [NeuS](https://github.com/Totoro97/NeuS). The main difference of these methods is in how the points along the ray are sampled and how the SDF is used during volume rendering. For more details of these methods, please check the corresponding paper. Here we explain these methods shortly and provide examples on how to use them in the following.

## UniSurf

UniSurf first finds the intersection of the surface and sample points around the surface. The sampling range starts from a large range and progressively decreases to a small range during training. When no surface is found for a ray, UniSurf samples uniformly according to the near and far value of the ray. To train a UniSurf model, run the following command:

```
ns-train unisurf --pipeline.model.sdf-field.inside-outside False sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65
```

## VolSDF

VolSDF uses an error-bound sampler [see paper for details] and converts the SDF value to a density value and then uses regular volume rendering as in NeRF. To train a VolSDF model, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.inside-outside False sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65
```

## NeuS

NeuS uses hierachical sampling with multiple steps and converts the SDF value to an alpha value based on a sigmoid function [see paper for details]. To train a NeuS model, run the following command:

```
ns-train neus --pipeline.model.sdf-field.inside-outside False sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65
```

## MonoSDF

[MonoSDF](https://github.com/autonomousvision/monosdf) builds on VolSDF and proposes to use monocular depth and normal cues as additional supervision. This is particularly helpful in sparse settings (little views) and in indoor scenes. To train a MonoSDF model for an indoor scene, run the following command:

```
ns-train monosdf --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include-mono-prior True

```

## Mono-UniSurf

Similar to MonoSDF, Mono-UniSurf uses monocular depth and normal cues as additional supervision for UniSurf. To train a Mono-UniSurf model, run the following command:

```
ns-train mono-unisurf --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include-mono-prior True
```

## Mono-NeuS

Similar to MonoSDF, Mono-NeuS uses monocular depth and normal cues as additional supervision for NeuS. To train a Mono-NeuS model, run the following command:

```
ns-train mono-neus --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include-mono-prior True
```

## Geo-NeuS

[Geo-NeuS](https://github.com/GhiXu/Geo-Neus) builds on NeuS and proposes a multi-view photometric consistency loss. To train a Geo-NeuS model on the DTU dataset, run the following command:

```
ns-train geo-neus --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan24 --load-pairs True
```

## Geo-UniSurf

The idea of Geo-NeuS can also applied to UniSurf, which we call Geo-UniSurf. To train a Geo-UniSurf model on the DTU dataset, run the following command:

```
ns-train geo-unisurf --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan24 --load-pairs True
```

## Geo-VolSDF

Similarly, we can apply the idea of Geo-NeuS to VolSDF, which we call Geo-VolSDF. To train a Geo-VolSDF model on the DTU dataset, run the following command:

```
ns-train geo-volsdf --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan24 --load-pairs True
```

## NeuS-acc

NeuS-acc maintains an occupancy grid for empty space skipping during point sampling along the ray. This significantly reduces the number of samples required during training and hence speeds up training. To train a NeuS-acc model on the DTU dataset, run the following command:

```
ns-train neus-acc --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan65

```

## NeuS-facto

NeuS-facto is inspired by [nerfacto](https://github.com/nerfstudio-project/nerfstudio) in nerfstudio, where the proposal network from [mip-NeRF360](https://jonbarron.info/mipnerf360/) is used for sampling points along the ray. We apply this idea to NeuS to speed up the sampling process and reduce the number of samples for each ray. To train a NeuS-facto model on the DTU dataset, run the following command:

```
ns-train neus-facto --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan65
```

## NeuralReconW

[NeuralReconW](https://github.com/zju3dv/NeuralRecon-W) is specifically designed for heritage scenes and hence can only be applied to these scenes. Specifically, it uses sparse point clouds from colmap to create a coarse occupancy grid. Using this occupancy grid, the near and far plane for each ray can be determined. Points are sampled uniformly along the ray within the near and far plane. Further, NeuralReconW also uses surface guided sampling, by sampling points in a small range around the predicted surface. To speed up sampling, it uses a high-resolution grid to cache the SDF field such that no network queries are required to find the surface intersection. The SDF cache is regularly updated during training (every 5K iterations). To train a NeuralReconW model on the DTU dataset, run the following command:

```
ns-train neusW --pipeline.model.sdf-field.inside-outside False heritage-data --data data/heritage/brandenburg_gate
```

# Representations

The representation stores geometry and appearance. The geometric mapping takes a 3D position as input and outputs an SDF value, a normal vector, and a geometric feautre vector. The color mapping (implemented as an MLP) takes a 3D position and view direction together with the normal vector and the geometry feature vector from the geometry mapping as input and outputs an RGB color vector.

We support three representations for the geometric mapping: MLPs, Multi-Res. Feature Grids from [iNGP](https://github.com/NVlabs/instant-ngp), and Tri-plane from [ConvONet](https://github.com/autonomousvision/convolutional_occupancy_networks) or [EG3D](https://github.com/NVlabs/eg3d). We now explain these representations in more detail:

## MLPs

The 3D position is encoded using a positional encoding as in NeRF and passed to a multi-layer perceptron (MLP) network to predict an SDF value, normal, and geometry feature. To train VolSDF with an MLP with 8 layers and 512 hidden dimensions, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.use-grid-feature sdfstudio-data --data YOUR_DATA
```

## Multi-res Feature Grids

The 3D position is first mapped to a multi-resolution feature grid, using tri-linear interpolation to retreive the corresponding feature vector. This feature vector is then used as input to an MLP to predict SDF, normal, and geometry features. To train a VolSDF model with Multi-Res Feature Grid representation with 2 layers and 256 hidden dimensions, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.encoding-type hash sdfstudio-data --data YOUR_DATA
```

## Tri-plane

The 3D position is first mapped to three orthogonal planes, using bi-linear interpolation to retreive a feature vector for each plane which are concatenated as as input to the MLP. To use a tri-plane representation on VolSDF, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature True  --pipeline.model.sdf-field.encoding-type tri-plane sdfstudio-data --data YOUR_DATA
```

## Geometry Initialization

Proper initialization is very important to obtain good results. By default, SDF Studio initializes the SDF as a sphere. For example, for the DTU dataset, you can initialize the network with the following command:

```
ns-train volsdf  --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.inside-outside False
```

For indoor scenes, please initialize the model using the following command:

```
ns-train volsdf --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.inside-outside True
```

Note that for indoor scenes the cameras are inside the sphere so we set `inside-outside` to `True` such that the points inside the sphere will have positive SDF values and points outside the sphere will have negative SDF values.

## Color Network

The color network is an MLPs, similar to the geometry MLP. It can be config using the following command:

```
ns-train volsdf --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim-color 512
```

# Supervision

## RGB Loss

We use the L1 loss for the RGB loss to supervise the volume rendered color at each ray. This is the default for all models.

## Mask Loss

The (optional) mask loss can be helpful to seperate the foreground object from the background. However, it requires additional masks as inputs. For example, in NeuralReconW, a segmentation network can be used to predict the sky region and the sky segmentation can be used as a label for the mask loss. The mask loss is used by default if masks are provided in the dataset. You can change the weight for the mask loss via:

```
--pipeline.model.fg-mask-loss-mult 0.001
```

## Eikonal Loss

The Eikonal loss is used for all SDF-based methods to regularize the SDF field to properly represent SDFs. It is not used for UniSurf which uses an occupancy field. You can change the weight of eikonal loss with the following command:

```
--pipeline.model.eikonal-loss-mult 0.01
```

## Smoothness Loss

The smoothness loss encourages smooth surfaces. This loss is used in UniSurf and encourages the normal of a surface point and the normal of a point sampled in its neighborhood to be similar. The weight for the smoothness loss can be changed with the following command:

```
--pipeline.model.smooth-loss-multi 0.01
```

## Monocular Depth Consistency

The monocular depth consistency loss is proposed in MonoSDF and uses depth predicted by a pretrained monocular depth network as additional constraint per image. This is particularly helpful in sparse settings (little views) and in indoor scenes. The weight for monocular depth consistency loss can be changed with the following command:

```
--pipeline.model.mono-depth-loss-mult 0.1
```

## Monocular Normal Consistency

The monocular normal consistency loss is proposed in MonoSDF and uses normals predicted by a pretrained monocular normal network as additional constraint during training. This is particularly helpful in sparse settings (little views) and in indoor scenes. The weight for monocular normal consistency loss can be changed with the following command:

```
--pipeline.model.mono-normal-loss-mult 0.05
```

## Multi-view Photometric Consistency

Encouraging multi-view photometric consistency is proposed in Geo-NeuS. For each ray, we seek the intersection with the surface and use the corresponding homography to warp patches from the source views to the target views and comparing those patches using normalized cross correlation (NCC). The weight for the multi-view photometric consistency loss can be changed with the following command:

```
--pipeline.model.patch-size 11 --pipeline.model.patch-warp-loss-mult 0.1 --pipeline.model.topk 4
```

where topk denotes the number of nearby views which have the smallest NCC error. Only those patchesare used for supervision, effectively ignoring outliers, e.g., due to occlusion.

## Sensor Depth Loss

RGBD data is useful for high-quality surface reconstruction. [Neural RGB-D Surface Reconstruction](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) propose two different loss functions: free space loss and sdf loss. Free space loss enforces the network to predict large SDF values between the camera origin and the truncation region of the observed surface. SDF loss enforces the network to predict approximate SDF values converted from depth observations. We further support L1 loss which enforce the consistency between volume rendered depth and sensor depth. The truncation value and the weights for sensor depth loss can be changed with the following command:

```bash
# truncation is set to 5cm with a rough scale value 0.3 (0.015 = 0.05 * 0.3)
--pipeline.model.sensor-depth-truncation 0.015 --pipeline.model.sensor-depth-l1-loss-mult 0.1 --pipeline.model.sensor-depth-freespace-loss-mult 10.0 --pipeline.model.sensor-depth-sdf-loss-mult 6000.0
```
