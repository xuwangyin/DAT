import dataclasses
import hashlib
import logging
import math
import os
import pathlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from sklearn import metrics
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from rebm.attacks.attack_steps import L2Step, LinfStep
from rebm.training.config_classes import ImageLogConfig
import torch.nn.functional as F
from torchmetrics.image.inception import InceptionScore
from PIL import Image



sys.path.insert(
    0, "pytorch-fid/src"
)
from pytorch_fid.fid_score import calculate_fid_given_paths

from rebm.attacks.adv_attacks import adam_attack, pgd_attack, pgd_attack_xent
from rebm.eval.inception_score import compute_inception_score

LOGGER = logging.getLogger(__name__)


def assert_no_grad(model):
    """
    Assert that all gradients in the model are either None or zeros.

    Args:
        model: PyTorch model to check

    Raises:
        AssertionError: If any parameter gradients are neither None nor zero
    """
    for p in model.parameters():
        if p.grad is not None and not torch.all(p.grad.eq(0)):
            raise AssertionError(
                "Some parameter gradients are neither None nor zero."
            )


def compute_img_diff(
    imgs_1: torch.Tensor, imgs_2: torch.Tensor
) -> torch.Tensor:
    # LOGGER.info(f"imgs_1: {imgs_1.shape}, imgs_2: {imgs_2.shape}")

    dif = imgs_2 - imgs_1
    dif_normalized = dif / (torch.max(torch.abs(dif)) + 1e-10)
    dif_scaled = (dif_normalized + 1) / 2
    return dif_scaled


def get_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    return metrics.auc(fpr, tpr)


def generate_images(
    attack_type: str,
    num_steps: int,
    model: nn.Module,
    x: torch.Tensor,
    attack_labels: torch.LongTensor | None = None,
    lr: float = None,
    eps: float = None,
    step_size: float = None,
    logsumexp=False,
    **kwargs,
):
    # If eps is 0.0, return the original images without modification
    if eps is not None and float(eps) == 0.0:
        return x.clone()
        
    if attack_type == "adam":
        imgs = adam_attack(
            model=model,
            x=x,
            lr=lr,
            steps=num_steps,
        )
    elif attack_type == "pgd":
        imgs = pgd_attack(
            model=model,
            x=x,
            attack_labels=attack_labels,
            norm="L2",
            eps=eps,  # infinity
            step_size=step_size,
            steps=num_steps,
            adv_targets=None,
            logsumexp=logsumexp,
        )
    else:
        raise NotImplementedError
    return imgs


def log_generate_images(
    cfg,
    model: nn.Module,
    samples: int = 10,
):
    if os.path.isfile(cfg.image_log.ood_data_dir):
        ood_samples = torch.load(cfg.image_log.ood_data_dir, weights_only=True)[
            :samples
        ]
        ood_dataset = torch.utils.data.TensorDataset(ood_samples)
    else:
        ood_dataset = torchvision.datasets.ImageFolder(
            cfg.image_log.ood_data_dir, transform=transforms.ToTensor()
        )
    loader = torch.utils.data.DataLoader(
        ood_dataset, batch_size=samples, num_workers=1, shuffle=False
    )
    ood_imgs = next(iter(loader))[0]

    attack_labels = torch.arange(cfg.data.num_classes).repeat(
        (samples // cfg.data.num_classes) + 1
    )[:samples]

    gen_imgs = generate_images(
        model=model,
        x=ood_imgs.to(cfg.device),
        attack_labels=attack_labels,
        logsumexp=False,
        **vars(cfg.image_log),
    )
    return gen_imgs


def evaluate_image_generation(model: nn.Module, cfg) -> tuple[float | None, torch.Tensor | None]:
    """Compute FID (if enabled) and sample images for logging."""
    assert_no_grad(model)
    fid, gen_imgs = None, None
    model.eval()

    if cfg.image_log.log_fid:
        if cfg.image_log.adaptive_steps:
            optimal_steps = find_optimal_steps(cfg, model)
            fid = compute_fid(
                model=model,
                cfg=cfg,
                override_fid_cfg={"num_steps": optimal_steps},
            )
        else:
            fid = compute_fid(
                model=model,
                cfg=cfg,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gen_imgs = log_generate_images(
        cfg=cfg,
        model=model,
        samples=10,
    )

    return fid, gen_imgs


def find_optimal_steps(cfg, model: nn.Module) -> int:
    if not cfg.image_log.adaptive_steps:
        return cfg.image_log.num_steps
    
    if cfg.image_log.step_sweep_range is not None:
        step_candidates = cfg.image_log.step_sweep_range
    else:
        base_steps = cfg.image_log.num_steps
        step_candidates = [max(1, base_steps + offset) for offset in [-4, -3, -2, -1, 0, 1, 2, 3, 4]]
    sweep_samples = cfg.image_log.sweep_num_samples
    
    LOGGER.info(f"Sweeping steps {step_candidates} with {sweep_samples} samples each")
    
    step_results = {}
    base_save_dir = cfg.image_log.save_dir
    
    for num_steps in step_candidates:
        step_save_dir = f"{base_save_dir}_sweep_{num_steps}steps"
        
        override_cfg = {
            'num_steps': num_steps,
            'num_samples': sweep_samples,
            'save_dir': step_save_dir
        }
        
        try:
            fid = compute_fid(cfg, model, override_fid_cfg=override_cfg, 
                            compute_is=False, save_visualization_grids=False)
            step_results[num_steps] = fid
            LOGGER.info(f"Steps {num_steps}: FID = {fid:.3f}")
        except Exception as e:
            LOGGER.warning(f"Failed to compute FID for {num_steps} steps: {e}")
            step_results[num_steps] = float('inf')
    
    if step_results:
        optimal_steps = min(step_results, key=step_results.get)
        optimal_fid = step_results[optimal_steps]
        LOGGER.info(f"Optimal steps: {optimal_steps} (FID: {optimal_fid:.3f})")
        return optimal_steps
    else:
        LOGGER.warning("No valid FID results, falling back to default steps")
        return cfg.image_log.num_steps


def compute_fid(
    cfg,
    model: nn.Module,
    override_fid_cfg: Optional[dict] = None,
    compute_is: bool = True,
    save_visualization_grids: bool = True,
):
    if override_fid_cfg is not None:
        fid_cfg_dict = {**dataclasses.asdict(cfg.image_log), **override_fid_cfg}
        fid_cfg = ImageLogConfig(**fid_cfg_dict)
    else:
        fid_cfg = cfg.image_log
        
    batch_size = cfg.batch_size
    num_workers = cfg.data.num_workers
    savedir = fid_cfg.save_dir
    
    # Add target class to save directory if specified
    if fid_cfg.target_class is not None:
        savedir = f"{savedir}_class{fid_cfg.target_class}"
    
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isfile(fid_cfg.ood_data_dir):
        ood_samples = torch.load(fid_cfg.ood_data_dir, weights_only=True)
        ood_dataset = torch.utils.data.TensorDataset(ood_samples)
    else:
        ood_dataset = torchvision.datasets.ImageFolder(
            fid_cfg.ood_data_dir, transform=transforms.ToTensor()
        )
    data_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    def save_image(image, idx):
        to_pil_image(image).save(
            os.path.join(savedir, f"{idx:06d}.{fid_cfg.img_extension}")
        )

    max_iters = math.ceil(fid_cfg.num_samples / batch_size)
    total_samples = fid_cfg.num_samples

    # Pre-generate all attack labels before the loop
    if fid_cfg.target_class is not None:
        # Use specified target class for all images
        all_attack_labels = torch.full(
            (total_samples,), fid_cfg.target_class, dtype=torch.long
        )
    else:
        # Generate deterministic pattern for all samples at once
        all_attack_labels = torch.arange(total_samples) % cfg.data.num_classes
        all_attack_labels = all_attack_labels.long()

    # Initialize dictionaries to accumulate samples by class
    max_samples_per_class = 10  # Default max samples to collect for each class
    num_classes_to_display = min(10, cfg.data.num_classes)  # Display at most 10 classes

    # Dictionaries to store samples by class
    gen_samples_by_class = {i: [] for i in range(num_classes_to_display)}
    seed_samples_by_class = {i: [] for i in range(num_classes_to_display)}

    for i, data in tqdm(
        zip(range(max_iters), data_loader),
        total=max_iters,
        desc=f"writting to {savedir}",
    ):
        seed_imgs = data[0] if isinstance(data, list) else data

        # Calculate how many samples we actually need for this batch
        samples_saved = i * batch_size
        remaining_samples = min(batch_size, total_samples - samples_saved)
        
        # Extract the labels for the current batch (limit to actual samples needed)
        start_idx = i * batch_size
        end_idx = start_idx + remaining_samples
        attack_labels = all_attack_labels[start_idx:end_idx]
        
        # Also limit seed_imgs to match attack_labels size
        seed_imgs = seed_imgs[:remaining_samples]

        gen_imgs = generate_images(
            model=model,
            x=seed_imgs.to(cfg.device),
            attack_labels=attack_labels,
            logsumexp=cfg.logsumexp_sampling,
            **vars(fid_cfg),
        )

        # Accumulate samples by class for visualization
        for j in range(gen_imgs.shape[0]):
            class_id = attack_labels[j].item()

            # Only accumulate samples for classes we want to display
            if class_id < num_classes_to_display:
                # Only add more samples if we haven't reached max_samples for this class
                if len(gen_samples_by_class[class_id]) < max_samples_per_class:
                    gen_samples_by_class[class_id].append(gen_imgs[j].cpu().clone())
                    seed_samples_by_class[class_id].append(seed_imgs[j].cpu().clone())

        # Save all images for FID calculation
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for j in range(remaining_samples):
                executor.submit(save_image, gen_imgs[j], samples_saved + j)

    if fid_cfg.target_class is not None:
        return 
    # Count actual generated samples
    actual_samples = len(list(pathlib.Path(savedir).glob(f'*.{fid_cfg.img_extension}')))
    LOGGER.info(f"Generated {actual_samples} samples (requested: {fid_cfg.num_samples})")

    # Create image grids from accumulated samples
    gen_grid_images = []
    seed_grid_images = []

    # Collect samples from each class
    for class_id in range(num_classes_to_display):
        samples = gen_samples_by_class[class_id]
        seed_samples = seed_samples_by_class[class_id]

        # Take up to max_samples_per_class samples
        gen_grid_images.extend(samples[:max_samples_per_class])
        seed_grid_images.extend(seed_samples[:max_samples_per_class])

    if save_visualization_grids and gen_grid_images:
        # Convert lists to tensors
        gen_grid_tensor = torch.stack(gen_grid_images)
        seed_grid_tensor = torch.stack(seed_grid_images)

        # Make grid with max_samples_per_class images per row
        grid = torchvision.utils.make_grid(
            gen_grid_tensor,
            nrow=max_samples_per_class,
            padding=1,
            normalize=True
        )

        # Make grid for seed images
        seed_grid = torchvision.utils.make_grid(
            seed_grid_tensor,
            nrow=max_samples_per_class,
            padding=1,
            normalize=True
        )

        # Create short filenames to avoid filesystem limits
        dir_hash = hashlib.md5(savedir.encode()).hexdigest()[:8]
        prefixed_gen_filename = f"gen_samples_{dir_hash}.png"
        prefixed_seed_filename = f"seed_samples_{dir_hash}.png"

        # Preserve the original path, but use the new prefixed filename
        gen_filepath = os.path.join(
            os.path.dirname(savedir), prefixed_gen_filename
        )
        seed_filepath = os.path.join(
            os.path.dirname(savedir), prefixed_seed_filename
        )

        # Save the grids with directory name as prefix
        to_pil_image(grid).save(gen_filepath)
        to_pil_image(seed_grid).save(seed_filepath)

        LOGGER.info(
            f"Created {num_classes_to_display}×{max_samples_per_class} grid visualizations at {gen_filepath} and {seed_filepath}"
        )

    fid = calculate_fid_given_paths(
        [savedir, fid_cfg.data_dir],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=num_workers,
    )
    
    if compute_is:
        is_mean, is_std = compute_inception_score(
            image_dir=savedir,
            batch_size=batch_size,
            splits=10,
            device=device
        )
        LOGGER.info(f"IS: {is_mean:.3f} ± {is_std:.3f}")
    
    def cleanup_directory(directory_path):
        try:
            shutil.rmtree(directory_path)
            LOGGER.info(f"Cleaned up generated images from {directory_path}")
        except Exception as e:
            LOGGER.warning(f"Failed to clean up directory {directory_path}: {e}")
    
    # Clean up generated images in background to avoid blocking training
    with ThreadPoolExecutor(max_workers=1) as cleanup_executor:
        cleanup_executor.submit(cleanup_directory, savedir)
    
    return fid


def compute_class_averaged_fid(
    cfg,
    model: nn.Module,
):
    fid_cfg = cfg.image_log
    batch_size = cfg.batch_size
    num_workers = cfg.data.num_workers
    num_classes = cfg.data.num_classes

    # Create class-specific directories for generated images
    class_savedirs = {}
    for class_idx in range(num_classes):
        class_savedirs[class_idx] = os.path.join(
            fid_cfg.save_dir, f"class_{class_idx}"
        )
        pathlib.Path(class_savedirs[class_idx]).mkdir(
            parents=True, exist_ok=True
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.isfile(fid_cfg.ood_data_dir):
        ood_samples = torch.load(fid_cfg.ood_data_dir, weights_only=True)
        ood_dataset = torch.utils.data.TensorDataset(ood_samples)
    else:
        ood_dataset = torchvision.datasets.ImageFolder(
            fid_cfg.ood_data_dir, transform=transforms.ToTensor()
        )
    data_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=1000,
        num_workers=num_workers,
        shuffle=False,
    )

    def save_image(image, class_idx, img_idx):
        to_pil_image(image).save(
            os.path.join(
                class_savedirs[class_idx],
                f"{img_idx:06d}.{fid_cfg.img_extension}",
            )
        )

    # Calculate samples per class
    samples_per_class = fid_cfg.num_samples // num_classes
    class_counters = {i: 0 for i in range(num_classes)}

    for data in tqdm(data_loader):
        seed_imgs = data[0] if isinstance(data, list) else data

        # Check which classes still need samples
        active_classes = [
            cls_idx
            for cls_idx, count in class_counters.items()
            if count < samples_per_class
        ]

        if not active_classes:
            break  # All classes have enough samples

        # Generate images for each class that still needs samples
        for class_idx in active_classes:
            # Skip if this class already has enough samples
            if class_counters[class_idx] >= samples_per_class:
                continue

            # Calculate how many more samples we need for this class
            samples_needed = min(
                batch_size, samples_per_class - class_counters[class_idx]
            )
            if samples_needed <= 0:
                continue

            # Use same class label for all images in batch
            attack_labels = torch.full(
                (samples_needed,), class_idx, dtype=torch.long
            )

            # Generate images with the specific class label
            gen_imgs = generate_images(
                model=model,
                x=seed_imgs[:samples_needed].to(cfg.device),
                attack_labels=attack_labels,
                **vars(fid_cfg),
            )

            # Save generated images with class-specific indices
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for j in range(samples_needed):
                    img_idx = class_counters[class_idx]
                    executor.submit(save_image, gen_imgs[j], class_idx, img_idx)
                    class_counters[class_idx] += 1

    # Verify that real data directory has class subdirectories
    real_data_root = fid_cfg.data_dir
    class_real_dirs = {}

    # Check if real data is organized by class
    if all(
        os.path.isdir(os.path.join(real_data_root, f"class_{i}"))
        for i in range(num_classes)
    ):
        # Real data is already organized by class
        for class_idx in range(num_classes):
            class_real_dirs[class_idx] = os.path.join(
                real_data_root, f"class_{class_idx}"
            )
    else:
        # Assume data is organized with class subdirectories (standard ImageFolder format)
        for class_idx in range(num_classes):
            class_real_dirs[class_idx] = os.path.join(
                real_data_root, str(class_idx)
            )

    # Calculate per-class FID scores
    fid_scores = []
    for class_idx in range(num_classes):
        try:
            class_fid = calculate_fid_given_paths(
                [class_savedirs[class_idx], class_real_dirs[class_idx]],
                batch_size=batch_size,
                device=device,
                dims=2048,
                num_workers=num_workers,
            )
            fid_scores.append(class_fid)
            print(f"FID for class {class_idx}: {class_fid}")
        except Exception as e:
            print(f"Error calculating FID for class {class_idx}: {e}")
            # Skip this class if there's an error

    # Calculate average FID
    avg_fid = sum(fid_scores) / len(fid_scores) if fid_scores else float("nan")
    print(f"Class-averaged FID: {avg_fid}")

    return avg_fid


def eval_acc(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the classification accuracy of a model on a dataloader.

    Args:
        model: The model to evaluate
        dataloader: DataLoader providing evaluation data
        device: Device to run evaluation on

    Returns:
        Accuracy as a float between 0 and 1
    """
    assert_no_grad(model)
    assert not model.training
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating accuracy"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, y=None)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0

    return correct / total


def eval_robust_acc(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    attack_fn: Callable = pgd_attack_xent,
    attack_kwargs: Dict = None,
    num_batches: Optional[int] = None,
    percentage: Optional[float] = None,
) -> float:
    """
    Measure the robust accuracy of a classification model against adversarial examples.

    Args:
        model: The model to evaluate
        dataloader: DataLoader providing evaluation data
        device: Device to run evaluation on
        attack_fn: Function to generate adversarial examples
        attack_kwargs: Additional arguments to pass to the attack function
        num_batches: Number of batches to evaluate (None for all)
        percentage: Percentage of dataset to evaluate (0-100), overrides num_batches if provided

    Returns:
        Robust accuracy as a float between 0 and 1
    """
    if attack_kwargs is None:
        # Roughly follow the settings in:
        # "Better Diffusion Models Further Improve Adversarial Training"
        # https://arxiv.org/pdf/2302.04638
        attack_kwargs = {
            "norm": "L2",
            "eps": 0.5,
            "step_size": 0.1,
            "steps": 10,
            "random_start": False,
        }

    assert_no_grad(model)
    assert not model.training

    # Calculate number of batches to evaluate based on percentage if provided
    if percentage is not None:
        if not 0 <= percentage <= 100:
            raise ValueError(
                f"Percentage must be between 0 and 100, got {percentage}"
            )
        total_batches = len(dataloader)
        num_batches = max(1, int(percentage * total_batches / 100))

    robust_correct = 0
    total_samples = 0
    batch_count = 0

    # Determine the total batches for tqdm
    total_iter = num_batches if num_batches is not None else len(dataloader)

    for inputs, labels in tqdm(
        dataloader, total=total_iter, desc="Evaluating robust accuracy"
    ):
        batch_count += 1
        if num_batches is not None and batch_count > num_batches:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)
        total_samples += batch_size

        # Generate adversarial examples
        adv_inputs = attack_fn(model, inputs, labels, **attack_kwargs)

        # Measure robust accuracy
        with torch.no_grad():
            adv_outputs = model(adv_inputs, y=None)
            _, adv_predicted = torch.max(adv_outputs.detach(), 1)
            robust_correct += (adv_predicted == labels).sum().item()

    # Calculate robust accuracy
    if total_samples == 0:
        return 0.0

    robust_accuracy = robust_correct / total_samples

    return robust_accuracy


def generate_outdist_adv_images(
    model, outdist_imgs, cfg, outdist_step, indist_labels=None
):
    """Generate adversarial images from out-of-distribution samples."""
    assert_no_grad(model)
    assert not model.training

    if indist_labels is not None:
        attack_labels = indist_labels
    else:
        assert False
        attack_labels = torch.randint(
            0,
            cfg.data.num_classes,
            (outdist_imgs.size(0),),
        )

    adv_imgs = generate_images(
        num_steps=outdist_step,
        model=model,
        x=outdist_imgs,
        attack_labels=attack_labels,
        logsumexp=cfg.logsumexp_sampling,
        **vars(cfg.attack),
    )
    assert not adv_imgs.requires_grad
    assert_no_grad(model)
    return adv_imgs, attack_labels


def generate_indist_adv_images(model, indist_imgs, indist_labels, cfg):
    """Generate adversarial images from in-distribution samples."""
    assert_no_grad(model)
    assert not model.training
    # Generate adversarial images from in-distribution samples
    assert indist_labels.dtype == torch.long

    # Generate attack labels for in-distribution samples that exclude their true labels
    # Make sure shifts is on the same device as indist_labels
    shifts = torch.randint(
        1,
        cfg.data.num_classes,
        (indist_imgs.size(0),),
        device=indist_labels.device,
    )
    indist_attack_labels = (indist_labels + shifts) % cfg.data.num_classes
    assert torch.all(indist_attack_labels != indist_labels)

    indist_adv_imgs = generate_images(
        num_steps=cfg.indist_attack.fixed_steps,
        model=model,
        x=indist_imgs,
        attack_labels=indist_attack_labels,
        **vars(cfg.indist_attack),
    )
    assert_no_grad(model)
    return indist_adv_imgs, indist_attack_labels


def generate_counterfactuals(model, train_loader, cfg):
    """
    Generates and saves counterfactual examples for the given model and data loader.
    Also computes FID scores and classifier confidence on counterfactual examples.

    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): DataLoader containing in-distribution images
        cfg (TrainConfig): Configuration parameters
    """
    import os
    import torchvision.utils as vutils
    import uuid
    import torch
    import torch.nn.functional as F
    from collections import defaultdict

    # Create a unique directory for this model's counterfactuals
    model_id = f"{pathlib.Path(cfg.config_path).stem}_{cfg.model.model_type}_{uuid.uuid4().hex[:8]}"
    save_dir = pathlib.Path(cfg.wandb_dir) / "counterfactuals" / model_id
    save_dir.mkdir(exist_ok=True, parents=True)

    LOGGER.info(f"Generating counterfactual examples to {save_dir}")

    # Get number of worker threads
    num_workers = (
        cfg.data.num_workers if hasattr(cfg.data, "num_workers") else 4
    )

    # Create subdirectories for each target class
    for class_idx in range(cfg.data.num_classes):
        (save_dir / f"{class_idx}").mkdir(exist_ok=True)

    # Define image saving function
    def save_image(image, batch_idx, img_idx, original_class, target_class):
        target_dir = save_dir / f"{target_class}"
        filename = f"batch{batch_idx:04d}_img{img_idx:04d}_original{original_class}.png"
        vutils.save_image(image, target_dir / filename)

    # Ensure model is in evaluation mode
    model.eval()

    # Dictionary to store confidence scores for each target class
    class_confidences = defaultdict(list)

    # Process batches from the data loader with tqdm progress bar
    for batch_idx, batch in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="Generating counterfactuals",
    ):
        # Extract images and labels
        indist_imgs = batch[0].to(cfg.device)
        indist_labels = batch[1].to(cfg.device)

        # Generate adversarial examples
        indist_adv_imgs, indist_attack_labels = generate_indist_adv_images(
            model=model,
            indist_imgs=indist_imgs,
            indist_labels=indist_labels,
            cfg=cfg,
        )

        # Get model predictions and confidence scores for adversarial images
        with torch.no_grad():
            logits = model(indist_adv_imgs, y=None)
            probabilities = F.softmax(logits, dim=1)

            # Store confidence scores for each target class
            for i in range(indist_adv_imgs.size(0)):
                target_class = indist_attack_labels[i].item()
                target_confidence = probabilities[i, target_class].item()
                class_confidences[target_class].append(target_confidence)

        # Save counterfactual images in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in range(indist_adv_imgs.size(0)):
                original_class = indist_labels[i].item()
                target_class = indist_attack_labels[i].item()
                executor.submit(
                    save_image,
                    indist_adv_imgs[i].cpu(),
                    batch_idx,
                    i,
                    original_class,
                    target_class,
                )

    # Compute per-class FID scores
    fid_scores = []
    confidence_scores = []
    reference_data_root = "./datasets/cifar10_images_by_classes/train/"

    # Ensure we're using CIFAR10 dataset
    assert "cifar10" in str(cfg.data.indist_dataset).lower(), (
        "Per-class FID calculation is currently only supported for CIFAR-10 dataset"
    )

    LOGGER.info("Computing per-class FID scores and confidence scores...")
    for class_idx in range(cfg.data.num_classes):
        class_gen_dir = str(save_dir / f"{class_idx}")
        class_ref_dir = os.path.join(reference_data_root, f"{class_idx}")

        # Count the number of samples in each directory
        gen_count = len(
            [f for f in os.listdir(class_gen_dir) if f.endswith(".png")]
        )
        ref_count = len(
            [f for f in os.listdir(class_ref_dir) if f.endswith(".png")]
        )

        # Calculate FID
        class_fid = calculate_fid_given_paths(
            [class_gen_dir, class_ref_dir],
            batch_size=cfg.batch_size,
            device=cfg.device,
            dims=2048,
            num_workers=num_workers,
        )
        fid_scores.append(class_fid)

        # Calculate average confidence for this class
        class_conf = sum(class_confidences[class_idx]) / len(class_confidences[class_idx]) if class_confidences[class_idx] else 0
        confidence_scores.append(class_conf)

        LOGGER.info(
            f"Class {class_idx}: FID = {class_fid:.2f}, Confidence = {class_conf:.4f} "
            f"(gen samples: {gen_count}, ref samples: {ref_count})"
        )

    # Calculate average FID and confidence
    avg_fid = sum(fid_scores) / len(fid_scores)
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    LOGGER.info(
        f"Counterfactual generation complete. Saved to {save_dir}. Average FID: {avg_fid:.2f}, Average confidence: {avg_confidence:.4f}"
    )

    return fid_scores, confidence_scores



def pgd_attack_uniform_target(
    model: nn.Module,
    x: torch.Tensor,
    norm: str,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Perform untargeted PGD attack using cross-entropy loss.

    Args:
        model: Model to attack
        x: Clean inputs
        norm: Norm to use for the attack ("L2" or "Linf")
        eps: Maximum perturbation size
        step_size: Step size for each iteration
        steps: Number of attack iterations
        random_start: Whether to start with random perturbation

    Returns:
        Adversarial examples that maximize the loss for the true labels
    """
    # Standard adversarial training uses pgd attack with model in training mode
    # assert not model.training
    assert not x.requires_grad

    if steps == 0:
        return x.clone()

    x0 = x.clone().detach()
    step_class = L2Step if norm == "L2" else LinfStep
    step = step_class(eps=eps, orig_input=x0, step_size=step_size)

    if random_start:
        x = step.random_perturb(x)

    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logits = model(x, y=None)

        num_classes = logits.size(1)
        uniform_targets = torch.full((logits.size(0), num_classes), 1.0 / num_classes).to(x.device)
        loss = F.kl_div(F.log_softmax(logits, dim=1), uniform_targets, reduction='none').sum(dim=1).sum()

        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=[x],
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=False,
        )
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x.clone().detach()

def ood_detection(model, indist_loader, outdist_loader, cfg):
    """
    Evaluate OOD detection capability by comparing in-distribution samples against both
    clean and adversarial out-of-distribution samples.

    Args:
        model (nn.Module): The neural network model
        indist_loader (DataLoader): DataLoader containing in-distribution images
        outdist_loader (DataLoader): DataLoader containing out-of-distribution images
        cfg (TrainConfig): Configuration parameters

    Returns:
        tuple: (clean_ood_auroc, adv_ood_auroc) AUROC scores for OOD detection
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from sklearn import metrics

    # Ensure model is in evaluation mode
    model.eval()
    assert_no_grad(model)

    # Initialize lists to store outputs
    indist_outputs = []
    ood_clean_outputs = []
    ood_adv_outputs = []

    # Maximum number of OOD samples to evaluate
    MAX_OOD_SAMPLES = 1024

    # Process in-distribution samples
    LOGGER.info("Processing in-distribution samples...")
    with torch.no_grad():
        for batch in tqdm(indist_loader, desc="In-distribution", disable=None):
            indist_imgs = batch[0].to(cfg.device)
            # Compute log-sum-exp of logits for in-distribution samples
            logits = model(x=indist_imgs, y=None)
            if cfg.ood_detection_logsumexp:
                batch_outputs = torch.logsumexp(logits, dim=1).cpu().numpy()
            else:
                batch_outputs = torch.max(F.softmax(logits, dim=1), dim=1)[0].cpu().numpy()
            indist_outputs.append(batch_outputs)

    # Concatenate all batch outputs
    indist_outputs = np.concatenate(indist_outputs)

    # Process out-of-distribution samples (limited to MAX_OOD_SAMPLES)
    LOGGER.info(f"Processing out-of-distribution samples (max {MAX_OOD_SAMPLES} samples)...")
    ood_samples_processed = 0

    for batch in tqdm(outdist_loader, desc="Out-of-distribution", disable=None):
        if ood_samples_processed >= MAX_OOD_SAMPLES:
            break
            
        ood_imgs = batch[0].to(cfg.device) if isinstance(batch, list) else batch.to(cfg.device)
        batch_size = ood_imgs.shape[0]
        
        # Determine how many samples to process from this batch
        samples_to_process = min(batch_size, MAX_OOD_SAMPLES - ood_samples_processed)
        
        # Only process the required number of samples from this batch
        if samples_to_process < batch_size:
            ood_imgs = ood_imgs[:samples_to_process]

        # Compute outputs for clean OOD samples
        with torch.no_grad():
            clean_logits = model(x=ood_imgs, y=None)
            if cfg.ood_detection_logsumexp:
                clean_batch_outputs = torch.logsumexp(clean_logits, dim=1).cpu().numpy()
            else:
                clean_batch_outputs = torch.max(F.softmax(clean_logits, dim=1), dim=1)[0].cpu().numpy()
            ood_clean_outputs.append(clean_batch_outputs)

        # Adjust eps and step_size based on image size (224x224 vs others)
        image_size = ood_imgs.shape[-1]
        if image_size == 224:
            if cfg.ood_detection_logsumexp:
                adv_ood_imgs = generate_images(
                    attack_type="pgd",
                    num_steps=10,
                    model=model,
                    x=ood_imgs,
                    attack_labels=None,
                    adv_targets=None,
                    logsumexp=True,
                    eps=3.0,
                    step_size=1.0
                )
            else:
                adv_ood_imgs = pgd_attack_uniform_target(
                    model=model,
                    x=ood_imgs,
                    norm="L2",
                    eps=3.0,
                    step_size=1.0,
                    steps=10,
                    random_start=False,
                )
        else:
            if cfg.ood_detection_logsumexp:
                adv_ood_imgs = generate_images(
                    attack_type="pgd",
                    num_steps=20,
                    model=model,
                    x=ood_imgs,
                    attack_labels=None,
                    adv_targets=None,
                    logsumexp=True,
                    eps=1.0,
                    step_size=0.1
                )
            else:
                adv_ood_imgs = pgd_attack_uniform_target(
                    model=model,
                    x=ood_imgs,
                    norm="L2",
                    eps=1.0,
                    step_size=0.1,
                    steps=20,
                    random_start=False,
                )

        # Compute log-sum-exp of logits for adversarial OOD samples
        with torch.no_grad():
            logits = model(x=adv_ood_imgs, y=None)
            if cfg.ood_detection_logsumexp:
                batch_outputs = torch.logsumexp(logits, dim=1).cpu().numpy()
            else:
                batch_outputs = torch.max(F.softmax(logits, dim=1), dim=1)[0].cpu().numpy()
            ood_adv_outputs.append(batch_outputs)

        # Update the count of processed samples
        ood_samples_processed += samples_to_process

    LOGGER.info(f"Processed {ood_samples_processed} OOD samples")

    # Concatenate all batch outputs
    ood_clean_outputs = np.concatenate(ood_clean_outputs)
    ood_adv_outputs = np.concatenate(ood_adv_outputs)

    # Compute AUROC for clean OOD detection
    clean_y_true = np.concatenate([np.ones_like(indist_outputs), np.zeros_like(ood_clean_outputs)])
    clean_y_score = np.concatenate([indist_outputs, ood_clean_outputs])
    clean_fpr, clean_tpr, _ = metrics.roc_curve(clean_y_true, clean_y_score)
    clean_auroc = metrics.auc(clean_fpr, clean_tpr)

    # Compute AUROC for adversarial OOD detection
    adv_y_true = np.concatenate([np.ones_like(indist_outputs), np.zeros_like(ood_adv_outputs)])
    adv_y_score = np.concatenate([indist_outputs, ood_adv_outputs])
    adv_fpr, adv_tpr, _ = metrics.roc_curve(adv_y_true, adv_y_score)
    adv_auroc = metrics.auc(adv_fpr, adv_tpr)

    return clean_auroc, adv_auroc
