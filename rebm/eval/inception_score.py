import logging
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def compute_inception_score(image_dir, batch_size=64, splits=10, device=None):
    """
    Compute Inception Score for images in a directory.

    Args:
        image_dir: Directory containing PNG images
        batch_size: Batch size for processing
        splits: Number of splits for IS calculation
        device: Device to use for computation

    Returns:
        tuple: (mean_score, std_score) of Inception Score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset of images from directory
    image_paths = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(".png")
        ]
    )

    if not image_paths:
        LOGGER.warning(f"No PNG images found in {image_dir}")
        return 0.0, 0.0

    # Transform images to required format for InceptionScore
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
        ]
    )

    # Create dataset
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)

    dataset = ImageDataset(image_paths, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize metric
    metric = InceptionScore(splits=splits).to(device)

    # Compute score
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Inception Score"):
            batch = batch.to(device)
            metric.update(batch)

    score = metric.compute()
    return score[0].item(), score[1].item()
