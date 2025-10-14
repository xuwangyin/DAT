# OpenImages Dataset Construction

This repository contains instructions for downloading the OpenImages dataset for training and validation.

## Dataset structure

The dataset should be organized as follows:
```
./
├── train/
│   └── data/
└── val/
    └── data/
```

## Download instructions

### Training dataset
1. Create the directory structure: `mkdir -p ./train/data/`
2. Download OpenImages files using the provided script:
   ```bash
   python minimal_openimages_downloader.py train_file_ids.txt ./train/data/ --section train
   ```

### Validation dataset
1. Create the directory structure: `mkdir -p ./val/data/`
2. Download OpenImages files using the provided script:
   ```bash
   python minimal_openimages_downloader.py val_file_ids.txt ./val/data/ --section train
   ```

## FID evaluation setup

After downloading the validation dataset, generate the files needed for FID evaluation:

```bash
python make_openimages_50K_val_fideval.py
```

This script processes (mainly resizes to 224x224) the validation images in `./val/data/` and saves the processed images to `fid_eval/data/` for FID evaluation.

## File ID lists

The file ID lists (`train_file_ids.txt` and `val_file_ids.txt`) contain the specific OpenImages file identifiers needed for dataset downloading.

## Dataset construction

The original script for constructing the OOD dataset by sampling from OpenImages training set (selecting samples with labels do not overlap with any ImageNet classes) is `create_ood_dataset.py`.
