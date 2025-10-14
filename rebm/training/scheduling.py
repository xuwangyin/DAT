"""Training event scheduling utilities."""


def get_lr_for_epoch(
    base_lr: float, epoch: int, total_epochs: int, dataset=None
) -> float:
    """
    Implements a stepwise learning rate decay with three phases:
    1. For the first 50% of training epochs, the learning rate remains at its maximum
    2. Between 50% and 75% of epochs, it decreases by a factor of 10
    3. In the final 25% of training, it further drops by a factor of 100

    Args:
        base_lr: The initial (maximum) learning rate
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs for training
        dataset: Optional dataset name for dataset-specific schedules

    Returns:
        The learning rate for the current epoch
    """
    if dataset in ["RestrictedImageNet", "ImageNet"]:
        assert total_epochs == 75
        if epoch < 30:
            return base_lr
        elif epoch < 60:
            return base_lr / 10.0
        elif epoch < 75:
            return base_lr / 100.0
        else:
            return base_lr / 1000.0
    else:
        if epoch > 200:
            # Adversarial Robustness on In- and Out-Distribution Improves Explainability
            return base_lr / 1000.0
        if epoch < total_epochs * 0.5:
            return base_lr
        elif epoch < total_epochs * 0.75:
            return base_lr / 10.0
        else:
            return base_lr / 100.0


def should_trigger_event(
    global_step_one_indexed: int,
    batch_size: int,
    interval_in_imgs: int,
    at_end: bool = False,
) -> bool:
    """Determine if a training event should trigger based on image count intervals.

    Args:
        global_step_one_indexed: Current training step (1-indexed)
        batch_size: Number of images per batch
        interval_in_imgs: Event interval in number of images
        at_end: If True, check if event should trigger at end of step;
                if False, check if event should trigger at start of step

    Returns:
        True if the event should trigger at this step

    Examples:
        >>> # Check if we should log metrics every 50000 images
        >>> should_trigger_event(step=100, batch_size=128, interval_in_imgs=50000)
        >>> # Check if we should save checkpoint at end of step
        >>> should_trigger_event(step=100, batch_size=128, interval_in_imgs=1000000, at_end=True)
    """
    global_step0 = global_step_one_indexed - 1
    global_images0 = global_step0 * batch_size
    next_images = (global_step0 + 1) * batch_size

    # Special case for first step
    if global_step0 == 0 and not at_end:
        return True

    if at_end:
        # Check if we're approaching the end of an interval
        current_interval = global_images0 // interval_in_imgs
        next_interval = next_images // interval_in_imgs
        return (current_interval < next_interval) or (
            global_images0 % interval_in_imgs >= interval_in_imgs - batch_size
        )
    else:
        # For start triggers, check if we're crossing into a new interval
        prev_interval = global_images0 // interval_in_imgs
        next_interval = next_images // interval_in_imgs
        return prev_interval < next_interval
