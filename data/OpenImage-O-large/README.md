# OpenImages Dataset Construction

This repository contains instructions for constructing the OpenImages dataset for training and validation.

## Dataset Structure

The dataset should be organized as follows:
```
./data/openimages/
├── train/
│   └── data/
└── val/
    └── data/
```

## Construction Instructions

### Training Dataset
1. Create the directory structure: `mkdir -p ./data/openimages/train/data/`
2. Download OpenImages files using the provided script:
   ```bash
   python minimal_openimages_downloader.py train_file_ids.txt ./data/openimages/train/data/ --section train
   ```

### Validation Dataset
1. Create the directory structure: `mkdir -p ./data/openimages/val/data/`
2. Download OpenImages files using the provided script:
   ```bash
   python minimal_openimages_downloader.py val_file_ids.txt ./data/openimages/val/data/ --section train
   ```

## FID Evaluation Setup

After downloading the validation dataset, generate the files needed for FID evaluation:

```bash
python make_openimages_50K_val_fideval.py
```

This script processes the validation images in `./data/openimages/val/data/` and saves the processed images to `OpenImagesO_fid_eval/data/` for FID evaluation.

## File ID Lists

The file ID lists (`train_file_ids.txt` and `val_file_ids.txt`) contain the specific OpenImages file identifiers needed for dataset construction.

## Notes

- Create the directory structure before downloading files
- Ensure sufficient disk space for the dataset
- Each line in the file ID lists corresponds to one OpenImages file to download