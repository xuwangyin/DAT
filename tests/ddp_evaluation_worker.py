"""Bounded real-process smoke test: CPU/Gloo by default, optional CUDA/NCCL."""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from rebm.training.average_model import AveragedModel
from rebm.training.distributed import (
    any_rank_requests_stop,
    run_rank_zero,
    wrap_training_model,
)
from rebm.training.evaluation import evaluation_model


def worker(rank, backend, rendezvous):
    torch.set_num_threads(1)
    os.environ["LOCAL_RANK"] = str(rank)
    device = (
        torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        model = wrap_training_model(nn.Linear(2, 2).to(device), device, True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        inputs = torch.ones(3, 2, device=device)
        calls = 0
        for use_ema in (False, True):
            ema = (
                AveragedModel(model.module, device=device) if use_ema else None
            )
            for _ in range(2):
                optimizer.zero_grad()
                model(inputs).sum().backward()
                optimizer.step()
                if ema is not None:
                    ema.update_parameters(model.module)

                def evaluate():
                    nonlocal calls
                    calls += 1
                    with evaluation_model(model, ema, True) as evaluating:
                        assert not isinstance(
                            evaluating,
                            (
                                nn.DataParallel,
                                nn.parallel.DistributedDataParallel,
                            ),
                        )
                        assert next(evaluating.parameters()).device == device
                        x = inputs.clone().requires_grad_(True)
                        evaluating(x).sum().backward()
                        assert x.grad is not None

                run_rank_zero(evaluate, use_ddp=True, device=device)
                assert model.training
                # Catch reducer/state problems in the first backward after eval.
                optimizer.zero_grad()
                model(inputs).sum().backward()
                optimizer.step()
        assert calls == (4 if rank == 0 else 0)
        weights = next(model.parameters()).detach()
        gathered = [torch.empty_like(weights) for _ in range(2)]
        dist.all_gather(gathered, weights)
        assert torch.equal(gathered[0], gathered[1])

        def failing_evaluation():
            with evaluation_model(model, None, True):
                raise ValueError("injected evaluation failure")

        try:
            run_rank_zero(failing_evaluation, use_ddp=True, device=device)
        except (ValueError, RuntimeError) as exc:
            assert "evaluation" in str(exc)
        else:
            raise AssertionError("evaluation failure did not reach this rank")
        assert model.training
        assert all(p.requires_grad for p in model.parameters())
        assert not any_rank_requests_stop(False, use_ddp=True, device=device)
        assert any_rank_requests_stop(rank == 1, use_ddp=True, device=device)
        dist.barrier()
        if rank == 0:
            print("two-rank evaluation passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--rendezvous", required=True, help="An unused absolute file path"
    )
    args = parser.parse_args()
    if args.backend == "nccl" and torch.cuda.device_count() < 2:
        parser.error("NCCL smoke test requires at least two visible CUDA GPUs")
    mp.spawn(worker, args=(args.backend, args.rendezvous), nprocs=2, join=True)
