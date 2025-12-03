# DAT

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

Model configuration files are located in `model_configs/` and specify the model architecture, training hyperparameters, and checkpoint paths for evaluation:

| Method | Configuration File |
|--------|-------------------|
| CIFAR-10 standard AT | `cifar10-at-WideResNet34x10.yaml` |
| CIFAR-10 RATIO | `cifar10-ratio-WideResNet34x10.yaml` |
| CIFAR-10 DAT (T=40) | `cifar10-dat-WideResNet34x10-T40-seed0.yaml` |
| CIFAR-10 DAT (T=50) | `cifar10-dat-WideResNet34x10-T50.yaml` |
| CIFAR-100 standard AT | `cifar100-at-WideResNet34x10.yaml` |
| CIFAR-100 RATIO | `cifar100-ratio-WideResNet34x10.yaml` |
| CIFAR-100 DAT (T=45) | `cifar100-dat-WideResNet34x10-T45-seed0.yaml` |
| CIFAR-100 DAT (T=50) | `cifar100-dat-WideResNet34x10-T50.yaml` |
| ImageNet standard AT ResNet50 | `imagenet-at-ResNet50ImageNet-256x256.yaml` |
| ImageNet standard AT WideResNet50x4 | `imagenet-at-WideResNet50x4ImageNet-256x256.yaml` |
| ImageNet standard AT ConvNeXtLarge | `imagenet-at-ConvNeXtLarge-convst-256x256.yaml` |
| ImageNet DAT ResNet50 (T=15) | `imagenet-dat-ResNet50ImageNet-T15-256x256.yaml` |
| ImageNet DAT ResNet50 (T=30) | `imagenet-dat-ResNet50ImageNet-T30-256x256.yaml` |
| ImageNet DAT WideResNet50x4 (T=30) | `imagenet-dat-WideResNet50x4ImageNet-T30-256x256.yaml` |
| ImageNet DAT WideResNet50x4 (T=65) | `imagenet-dat-WideResNet50x4ImageNet-T65-256x256.yaml` |
| ImageNet DAT ConvNeXtLarge | `imagenet-dat-ConvNeXtLarge-convst-256x256-stepsize3_lr0.0003.yaml` |

## Model Training

To train models, use the following command:

```bash
python train.py $config_file
```

For example:
```bash
python train.py model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml
```

## Model Evaluation

- **Classification Accuracy**: Evaluate standard and robust accuracy:
  ```bash
  python run_acc_eval.py $config_file
  ```
  Example: `python run_acc_eval.py model_configs/cifar10-dat-WideResNet34x10-T50.yaml`

- **Fréchet Inception Distance (FID)**: Evaluate the quality of generated images using FID:
  ```bash
  python run_fid_eval.py $config_file
  ```
  Example: `python run_fid_eval.py model_configs/cifar10-dat-WideResNet34x10-T50.yaml`

  **Precision & Recall**: For additional P&R metrics using OpenAI's official evaluator:

  One-time setup:
  ```bash
  # Clone evaluator repository
  git clone https://github.com/openai/guided-diffusion.git
  # Download ImageNet 256×256 reference batch
  wget -P ./guided-diffusion/evaluations https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
  ```

  Usage:
  ```bash
  # Generate samples and keep them
  python run_fid_eval.py $config_file --keep-samples
  # Convert images to npz format
  python images_to_npz.py image_log/<run_id> image_log/<run_id>.npz
  # Run OpenAI's evaluator
  python ./guided-diffusion/evaluations/evaluator.py ./guided-diffusion/evaluations/VIRTUAL_imagenet256_labeled.npz image_log/<run_id>.npz
  ```

- **Expected Calibration Error (ECE)**: Evaluate the calibration quality of model predictions:
  ```bash
  python run_acc_eval.py --calibration $config_file
  ```

- **Out-of-Distribution (OOD) Detection**: Evaluate how well models detect out-of-distribution inputs:
  ```bash
  python run_ood_eval.py $config_file --ood-dataset $ood_dataset
  ```
  Example: `python run_ood_eval.py model_configs/cifar100-at-WideResNet34x10.yaml --ood-dataset noise`

## Related Work

- **EGC Adversarial Robustness Evaluation**: [https://github.com/xuwangyin/EGC-robustness](https://github.com/xuwangyin/EGC-robustness) - Repository for evaluating the adversarial robustness of EGC ([arXiv:2304.02012](https://arxiv.org/abs/2304.02012))

- **SADA-JEM Robustness Evaluation**: [https://github.com/xuwangyin/SADAJEM-robustness](https://github.com/xuwangyin/SADAJEM-robustness) - Repository for evaluating the robustness of SADA-JEM ([arXiv:2209.07959](https://arxiv.org/abs/2209.07959))
