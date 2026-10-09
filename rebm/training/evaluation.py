"""Evaluation model ownership and state restoration (no dataset dependencies)."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rebm.training.average_model import AveragedModel


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return model weights without assuming a particular DP/DDP/EMA nesting."""
    while isinstance(model, (nn.DataParallel, DDP, AveragedModel)):
        model = model.module
    return model


@contextmanager
def evaluation_model(
    model: nn.Module, ema_model: nn.Module | None, use_ddp: bool
):
    """Borrow the local model for evaluation and restore its state on exit.

    DDP evaluation must bypass the DDP forward and must not create DataParallel
    replicas on GPUs owned by other ranks. Non-DDP runs retain DataParallel.
    Keep autograd enabled: DAT sampling and robust accuracy need input gradients.
    """
    source = ema_model if ema_model is not None else model
    modes = [(module, module.training) for module in source.modules()]
    parameters = [
        (parameter, parameter.requires_grad)
        for parameter in source.parameters()
    ]
    eval_model = unwrap_model(source) if use_ddp else source
    if not use_ddp and ema_model is not None:
        eval_model = nn.DataParallel(source)
    device = next(eval_model.parameters()).device
    try:
        eval_model.eval()
        eval_model.zero_grad(set_to_none=True)
        # Sampling only needs input gradients; do not accumulate parameter grads
        # (in particular in the Adam sampler) or invoke DDP reducer hooks.
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield eval_model
    finally:
        eval_model.zero_grad(set_to_none=True)
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
