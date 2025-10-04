"""Training event scheduling utilities."""


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
