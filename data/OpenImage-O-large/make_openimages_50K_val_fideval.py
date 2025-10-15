import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


def get_imagenet256_dataset(datadir, interpolation=2, transform=None):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    print("imagenet transform:")
    print(transform)
    return torchvision.datasets.ImageFolder(datadir, transform)


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

ood_dataset = get_imagenet256_dataset(datadir="data/openimages/val")
print(f"Dataset size: {len(ood_dataset)} samples")

batch_size = 100
loader = torch.utils.data.DataLoader(
    ood_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(0),
    num_workers=8,
)

savedir = "OpenImagesO_fid_eval/data"
max_iters = 50000 // batch_size
samples_saved = 0


def save_image(image, idx):
    to_pil_image(image).save(os.path.join(savedir, f"{idx:06d}.png"))


pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
for i, data in tqdm(zip(range(max_iters), loader), total=max_iters):
    seed_imgs = data[0] if isinstance(data, list) else data
    with ThreadPoolExecutor(max_workers=20) as executor:
        for j in range(seed_imgs.shape[0]):
            executor.submit(save_image, seed_imgs[j], samples_saved)
            samples_saved += 1

print(f"Saved {samples_saved} samples to {savedir}")
