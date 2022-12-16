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

[NeuralReconW](https://github.com/zju3dv/NeuralRecon-W) is specifically designed for heritage scenes and hence can only be applied to these scenes. Specifically, it uses sparse point clouds from colmap to create a coarse occupancy grid. Using this occupancy grid, the near and far plane for each ray can be determined. Points are sampled uniformly along the ray within the near and far plane. Further, NeuralReconW also uses surface guided sampling, by sampling points in a small range around the predicted surface. To speed up sampling, it uses a high-resolution grid to cache the SDF field such that no network queries are required to find the surface intersection. The SDF cache is regularly updated during training (e.g. every 5K iterations). To train a NeuralReconW model on the DTU dataset, run the following command:

```
ns-train neusW --pipeline.model.sdf-field.inside-outside False heritage-data --data data/heritage/brandenburg_gate
```

# Representations

The neural representation contains two parts, a geometric network and a color network. The geometric network takes a 3D position as input and outputs a sdf value, a normal vector, and a geometric feautre vector. The color network takes a 3D position and view direction together with the normal vector and the geometric feautre vector from geometric network and as inputs and outputs a RGB color vector.

We support three representations for the geometric network: MLPs, [Multi-Res. Feature Grids](https://github.com/NVlabs/instant-ngp), and [Tri-plane](https://github.com/apchenstu/TensoRF). We now explain the details and how to use it in the following:

## MLPs

The 3D position is encoded with positional encoding as in nerf and pass to a multi-layer perception network to prediction sdf, normal, and geometric feature. For example, to train VolSDF with a MLPs with 8 layers and 512 hiddin dimension, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.use-grid-feature sdfstudio-data --data YOUR_DATA
```

## Multi-res feature grids

The 3D position is first mapped to a multi-resolution feature grids and use tri-linear interpolation to retreive the corresponding feature vector. The feature vector is used as input to a MLPs to prediction sdf, normal, and geometric feature. For example, to use a VolSDF model with Multi-Res Feature Grids representations with 2 layers and 256 hiddin dimension, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.encoding-type hash sdfstudio-data --data YOUR_DATA
```

## Tri-plane

The 3D position is first mapped to three orthogonal planes and use bi-linear interpolation to retreive feature vector for each plane and concat them as input the the MLPs. To use tri-plane representation on VolSDF, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature True  --pipeline.model.sdf-field.encoding-type tri-plane sdfstudio-data --data YOUR_DATA
```

## Geometric initilaization

Good initialization is important to get good results. So we usually initialize the SDF as a sphere. For example, in the DTU dataset, we usually initialize the network with the following command: 

```
ns-train volsdf  --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.inside-outside False
```

And in the indoor scene we use the initilization with the following command:

```
ns-train volsdf --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.inside-outside True
```

Note that in the indoor scenes, cameras are inside the sphere so we set `inside-outside` to `True` such that the points inside the sphere will have positive SDF value and points outside the sphere will have negetive SDF value.

## Color network

The color netwokr is a MLPs, similar to geometric network, it can be config with the following command:
```
ns-train volsdf --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim-color 512
```

# Supervision

## RGB loss

We use L1 loss for the RGB loss to supervise the volume rendered color for each ray. It is always used for all models.

## Mask loss

The mask loss is usually helpful to seperate foreground object and background. However, it needs additonal inputs. For example, in neuralreconW, a segmentation network is used to predict the sky region and the sky segmentation is used as a label for mask loss. It is used by default if masks are provided in the dataset. You could change the weight for the mask loss with
```
--pipeline.model.fg-mask-loss-mult 0.001
```

## Eikonal loss

Eikonal loss is used in all SDF-based method to regularize the SDF field except UniSurf because UniSurf uses occupancy field. You could change the weight of eikonal loss with the following command:
```
--pipeline.model.eikonal-loss-mult 0.01
```

## Smoothness loss

The smoothness loss enforces smoothness surface. It is used in UniSurf and the weight for the smoothness loss can be changed with the following command:
```
--pipeline.model.smooth-loss-multi 0.01
```

## Monocular depth consistency

The monocular depth consistency loss is proposed in MonoSDF which use a pretrained monocular depth network to provided priors during training. This is particularly helpful in sparse settings (little views) and in indoor scenes. The weight for monocular depth consistency loss can be changed with the following command:
```
--pipeline.model.mono-depth-loss-mult 0.1
```

## Monocular normal consistency
The monocular normal consistency loss is proposed in MonoSDF which use a pretrained monocular normal network to provided priors during training. This is particularly helpful in sparse settings (little views) and in indoor scenes. The weight for monocular normal consistency loss can be changed with the following command:
```
--pipeline.model.mono-normal-loss-mult 0.05
```

## Multi-view photometric consistency

Multi-view photometric consistency is proposed in Geo-NeuS to enforce multi-view geometry consistency. For each ray, it finds the intersection with the surface and uses homography to warp patches from source views to target view and uses normalized cross correaltion loss (NCC) for warped patches. The weight for multi-view photometric consistency can be changed with the following command:
```
ns-train volsdf --pipeline.model.patch-size 11 --pipeline.model.patch-warp-loss-mult 0.1 --pipeline.model.topk 4
```
where topk is number of nearby views that have smalleast NCC error and only these patches with lowest error are used for supervision. It is an approximation of occlusion handling. 
