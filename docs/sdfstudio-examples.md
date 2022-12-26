# Reproduce project page results

Here, we provide commands to reproduce reconstruction results on our project page. Please download the corresponding dataset before you run the following commands.

## NeuS-facto on the heritage dataset

```bash
ns-train neus-facto-bigmlp --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False --pipeline.model.sdf-field.bias 0.3 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.model.eikonal-loss-mult 0.0001 --pipeline.model.num-samples-outside 4 --pipeline.model.background-model grid --trainer.steps-per-eval-image 5000 --vis wandb --experiment-name neus-facto-bigmlp-gate --machine.num-gpus 8 heritage-data --data data/heritage/brandenburg_gate
```

## Unisurf, VolSDF, and NeuS with multi-res. grids on the DTU dataset

```bash
# unisurf
ns-train unisurf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --trainer.steps-per-eval-image 5000 --pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.model.background-model none --vis wandb --experiment-name unisurf-dtu122  sdfstudio-data --data data/dtu/scan122

# volsdf
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.beta-init 0.1 --trainer.steps-per-eval-image 5000 --pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.model.background-model none --vis wandb --experiment-name volsdf-dtu106  sdfstudio-data --data data/dtu/scan106

# neus
ns-train neus --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.beta-init 0.3 --trainer.steps-per-eval-image 5000 --pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.model.background-model none --vis wandb --experiment-name neus-dtu114  sdfstudio-data --data data/dtu/scan114
```

## Geo-Unisurf, Geo-VolSDF, and Geo-NeuS with MLP on the DTU dataset

```bash
# geo-unisurf
ns-train geo-unisurf --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.num-layers 8 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name geo-unisurf-dtu110 --pipeline.datamanager.train-num-rays-per-batch 4096 sdfstudio-data --data data/dtu/scan110 --load-pairs True

# geo-volsdf
ns-train geo-volsdf --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.num-layers 8 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.beta-init 0.1 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name geo-volsdf-dtu97 --pipeline.model.eikonal-loss-mult 0.1 --pipeline.datamanager.train-num-rays-per-batch 4096 sdfstudio-data --data data/dtu/scan97 --load-pairs True

#geo-neus
ns-train geo-neus --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.num-layers 8 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name geo-volsdf-dtu24 --pipeline.model.eikonal-loss-mult 0.1 --pipeline.datamanager.train-num-rays-per-batch 4096 sdfstudio-data --data data/dtu/scan24 --load-pairs True

```

## MonoSDF on the Tanks and Temples dataset

```bash
ns-train monosdf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True  --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.1 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name monosdf-htnt-scan1 --pipeline.model.mono-depth-loss-mult 0.001 --pipeline.model.mono-normal-loss-mult 0.01 --pipeline.datamanager.train-num-rays-per-batch 2048 --machine.num-gpus 8 sdfstudio-data --data data/tanks-and-temple-highres/scan1 --include_mono_prior True --skip_every_for_val_split 30
```

## NeuS-facto-bigmlp on the Tanks and Temples dataset with monocular prior (Mono-NeuS)

```bash
ns-train neus-facto-bigmlp --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True  --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name neus-facto-bigmlp-tnt2 --pipeline.model.mono-depth-loss-mult 0.1 --pipeline.model.mono-normal-loss-mult 0.05 --pipeline.model.eikonal-loss-mult 0.01 --pipeline.datamanager.train-num-rays-per-batch 4096 --machine.num-gpus 8 sdfstudio-data --data data/tanks-and-temple/scan2 --include_mono_prior True --skip_every_for_val_split 30
```

## NeuS-acc with monocular prior on the Replica dataset

```bash
ns-train neus-acc --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.model.eikonal-loss-mult 0.1 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --pipeline.model.mono-depth-loss-mult 0.1 --pipeline.model.mono-normal-loss-mult 0.05 --pipeline.datamanager.train-num-rays-per-batch 2048 --vis wandb --experiment-name neus-acc-replica1 sdfstudio-data --data data/replica/scan1 --include_mono_prior True
```

## NeuS-RGBD on the synthetic Neural-rgbd dataset

```bash
#kitchen
ns-train neus --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True  --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.datamanager.train-num-images-to-sample-from -1 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name kitchen_sensor_depth-neus --pipeline.model.sensor-depth-l1-loss-mult 0.1 --pipeline.model.sensor-depth-freespace-loss-mult 10.0 --pipeline.model.sensor-depth-sdf-loss-mult 6000.0 --pipeline.model.mono-normal-loss-mult 0.05 --pipeline.datamanager.train-num-rays-per-batch 2048 --machine.num-gpus 1 sdfstudio-data --data data/neural_rgbd/kitchen_sensor_depth --include_sensor_depth True --skip_every_for_val_split 30

# breadfast-room
ns-train neus --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True  --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.datamanager.train-num-images-to-sample-from -1 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name breakfast_room_sensor_depth-neus --pipeline.model.sensor-depth-l1-loss-mult 0.1 --pipeline.model.sensor-depth-freespace-loss-mult 10.0 --pipeline.model.sensor-depth-sdf-loss-mult 6000.0 --pipeline.model.mono-normal-loss-mult 0.05 --pipeline.datamanager.train-num-rays-per-batch 2048 --machine.num-gpus 1 sdfstudio-data --data data/neural_rgbd/breakfast_room_sensor_depth --include_sensor_depth True --skip_every_for_val_split 30
```
