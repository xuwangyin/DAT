"""Device ownership and rank-zero work shared by training and evaluation."""

import os
from collections.abc import Callable
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def process_device(use_ddp: bool) -> torch.device:
    """Resolve the node-local CUDA ordinal; a global rank is never a device ID."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if not use_ddp:
        return torch.device("cuda:0")
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "CUDA DDP requires LOCAL_RANK (launch with torchrun)"
        ) from exc
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside the visible CUDA devices"
        )
    return torch.device("cuda", local_rank)


def initialize_distributed(backend: str) -> torch.device:
    """Bind the local device before initializing any NCCL collectives."""
    device = process_device(use_ddp=True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif backend == "nccl":
        raise ValueError("NCCL training requires CUDA devices")
    if not dist.is_initialized():
        # Rank-zero FID can be slow. This is still a finite timeout, not a
        # guarantee that arbitrarily long evaluation will succeed.
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=90))
    return device


def wrap_training_model(model: nn.Module, device: torch.device, use_ddp: bool):
    if use_ddp:
        # Disabling buffer broadcast preserves the existing DAT training policy.
        # The global rank can exceed the number of GPUs on this node.
        device_ids = [device.index] if device.type == "cuda" else None
        return DDP(model, device_ids=device_ids, broadcast_buffers=False)
    return nn.DataParallel(model)


def run_rank_zero(
    callback: Callable[[], None], *, use_ddp: bool, device: torch.device
) -> None:
    """Run collective-free work on rank zero while every rank waits.

    All ranks must call this at the same point. The entry barrier keeps peers
    out of training, and the completion broadcast releases them only after the
    callback (including resource cleanup) finishes. Ordinary Python failures
    abort all ranks instead of allowing peers to enter the next DDP forward.
    Process death, CUDA faults, and work exceeding the process-group timeout
    still require the launcher's failure handling.
    """
    if not use_ddp:
        callback()
        return

    dist.barrier()
    # Gloo communicates the status on CPU; NCCL requires the owned CUDA device.
    status_device = (
        device if dist.get_backend() == "nccl" else torch.device("cpu")
    )
    status = torch.zeros(1, dtype=torch.int, device=status_device)
    error = None
    if dist.get_rank() == 0:
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 - broadcast failure, then re-raise
            error = exc
            status.fill_(1)
    dist.broadcast(status, src=0)
    if status.item():
        if error is not None:
            raise error
        raise RuntimeError(
            "Rank 0 evaluation/checkpoint work failed; see rank 0 traceback"
        )


def any_rank_requests_stop(
    requested: bool, *, use_ddp: bool, device: torch.device
) -> bool:
    """Make a signal received by any rank a collective decision to stop."""
    if not use_ddp:
        return requested
    status_device = (
        device if dist.get_backend() == "nccl" else torch.device("cpu")
    )
    flag = torch.tensor([int(requested)], device=status_device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())
