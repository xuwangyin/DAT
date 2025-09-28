# Dual-AT

## Setup Environment

Create the conda environment using micromamba:

```bash
micromamba env create -f packages.yaml
```

## Setup Data

- **OpenImages-O-large**: Setup according to `data/OpenImage-O-large/README.md`
- **ImageNet**: Setup ImageNet dataset to `data/ImageNet/`, which should contain: `ILSVRC2012_devkit_t12.tar.gz`, `test/`, `train/`, `val/`
- **Tiny Images**: Download [tiny_images.bin](http://www.archive.org/download/80-million-tiny-images-2-of-2/tiny_images.bin) to `./data/`
- **Model checkpoints and FID data**: Download the folders under [this Dropbox folder](https://www.dropbox.com/scl/fo/81n3lxw5dvhds56guuegh/AN3sYMNamKQnCkkFXC3CKcg?rlkey=ih9m3rasrmuzeyxwe7qjcpue9&st=b3a1r2nz&dl=0) and place them under the project directory

## Model Configuration Files

The following table outlines the configuration files in `model_configs/` and their purposes:

| Method | Configuration File |
|--------|-------------------|
| CIFAR-10 standard AT | `cifar10-at-WideResNet34x10.yaml` |
| CIFAR-10 RATIO | `cifar10-ratio-WideResNet34x10.yaml` |
| CIFAR-10 Dual-AT (T=40) | `cifar10-dual-at-WideResNet34x10-T40-seed0.yaml` |
| CIFAR-10 Dual-AT (T=50) | `cifar10-dual-at-WideResNet34x10-T50.yaml` |
| CIFAR-100 standard AT | `cifar100-at-WideResNet34x10.yaml` |
| CIFAR-100 RATIO | `cifar100-ratio-WideResNet34x10.yaml` |
| CIFAR-100 Dual-AT (T=45) | `cifar100-dual-at-WideResNet34x10-T45-seed0.yaml` |
| CIFAR-100 Dual-AT (T=50) | `cifar100-dual-at-WideResNet34x10-T50.yaml` |
| ImageNet standard AT ResNet50 | `imagenet-at-ResNet50ImageNet.yaml` |
| ImageNet standard AT WideResNet50x4 | `imagenet-at-WideResNet50x4ImageNet.yaml` |
| ImageNet Dual-AT ResNet50 (T=15) | `imagenet-dual-at-ResNet50ImageNet-T15-300K-seed0.yaml` |
| ImageNet Dual-AT ResNet50 (T=30) | `imagenet-dual-at-ResNet50ImageNet-T30.yaml` |
| ImageNet Dual-AT WideResNet50x4 (T=30) | `imagenet-dual-at-WideResNet50x4ImageNet-T30.yaml` |
| ImageNet Dual-AT WideResNet50x4 (T=65) | `imagenet-dual-at-WideResNet50x4ImageNet-T65.yaml` |

## Model Training

To train models, use the following command:

```bash
python train.py $config_file
```

For example:
```bash
python train.py model_configs/cifar10-dual-at-WideResNet34x10-T40-seed0.yaml
```

## Model Evaluation

- **Classification Accuracy**: Evaluate standard and robust accuracy:
  ```bash
  python eval_acc.py $config_file
  ```
  Example: `python eval_acc.py model_configs/cifar10-dual-at-WideResNet34x10-T50.yaml`

- **Fréchet Inception Distance (FID)**: Evaluate the quality of generated images using FID:
  ```bash
  python eval_fid.py $config_file
  ```
  Example: `python eval_fid.py model_configs/cifar10-dual-at-WideResNet34x10-T50.yaml`

- **Expected Calibration Error (ECE)**: Evaluate the calibration quality of model predictions:
  ```bash
  python eval_acc.py --calibration $config_file
  ```

- **Out-of-Distribution (OOD) Detection**: Evaluate how well models detect out-of-distribution inputs:
  ```bash
  python eval_ood_detection.py $config_file --ood-dataset $ood_dataset
  ```
  Example: `python eval_ood_detection.py model_configs/cifar100-at-WideResNet34x10.yaml --ood-dataset noise`

## Related Work

- **EGC Adversarial Robustness Evaluation**: [https://github.com/xuwangyin/EGC-robustness](https://github.com/xuwangyin/EGC-robustness) - Repository for evaluating the adversarial robustness of EGC ([arXiv:2304.02012](https://arxiv.org/abs/2304.02012))

- **SADA-JEM Robustness Evaluation**: [https://github.com/xuwangyin/SADAJEM-robustness](https://github.com/xuwangyin/SADAJEM-robustness) - Repository for evaluating the robustness of SADA-JEM ([arXiv:2209.07959](https://arxiv.org/abs/2209.07959))
