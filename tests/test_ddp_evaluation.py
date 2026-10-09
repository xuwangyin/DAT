"""CPU regression coverage; no datasets, weights, CUDA, or wandb service needed."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from rebm.training import distributed
from rebm.training.average_model import AveragedModel
from rebm.training.config_classes import load_train_config
from rebm.training.evaluation import evaluation_model, unwrap_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments/cifar10/cifar10_conditional_r1reg0_augauto_innoutloader_steps40_bs128_indistxent_sgd_lr0.001_stesp50_wd_wrn3410_indistaugoriginal_innoutpretrained_mean0_bestavg_ema_logsumexp_run2.yaml"
)


@pytest.fixture
def cfg():
    return load_train_config(
        str(CONFIG),
        [
            "use_ddp=True",
            "batch_size=8",
            "n_imgs_per_evaluation_log=16",
            "data.num_workers=0",
            "robust_eval=False",
            "wandb_disabled=True",
        ],
    )


@pytest.fixture
def training():
    # Integration tests import the real training module. Only external I/O and
    # expensive evaluators are mocked, not the code under test.
    import rebm.training.train as training

    return training


def test_cuda_device_uses_local_rank_before_initialization(monkeypatch, cfg):
    events = []
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda d: events.append(("device", d))
    )
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed.dist,
        "init_process_group",
        lambda **kw: events.append(("init", kw)),
    )
    device = distributed.initialize_distributed("nccl")
    assert device == cfg.device == torch.device("cuda:1")
    assert [event[0] for event in events] == ["device", "init"]
    wrapper = Mock()
    monkeypatch.setattr(distributed, "DDP", wrapper)
    model = nn.Linear(2, 2)
    distributed.wrap_training_model(model, device, use_ddp=True)
    wrapper.assert_called_once_with(
        model, device_ids=[1], broadcast_buffers=False
    )


@pytest.mark.parametrize("local_rank", [None, "invalid", "-1", "2"])
def test_invalid_local_rank_fails_early(monkeypatch, local_rank):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("RANK", "7")
    if local_rank is None:
        monkeypatch.delenv("LOCAL_RANK", raising=False)
    else:
        monkeypatch.setenv("LOCAL_RANK", local_rank)
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        distributed.process_device(True)


def test_cpu_and_non_ddp_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert distributed.process_device(True) == torch.device("cpu")
    with pytest.raises(ValueError, match="NCCL"):
        distributed.initialize_distributed("nccl")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("LOCAL_RANK", "7")
    assert distributed.process_device(False) == torch.device("cuda:0")


@pytest.mark.parametrize(
    "rank,failed", [(0, False), (1, False), (0, True), (1, True)]
)
def test_rank_zero_collective_order_and_failure(monkeypatch, rank, failed):
    events = []
    monkeypatch.setattr(distributed.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        distributed.dist, "barrier", lambda: events.append("barrier")
    )

    def callback():
        events.append("callback")
        if failed:
            raise ValueError("injected failure")

    def broadcast(status, src):
        events.append("broadcast")
        assert src == 0
        if rank == 0:
            assert bool(status.item()) == failed
        else:
            status.fill_(int(failed))

    monkeypatch.setattr(distributed.dist, "broadcast", broadcast)
    if failed:
        with pytest.raises(ValueError if rank == 0 else RuntimeError):
            distributed.run_rank_zero(
                callback, use_ddp=True, device=torch.device("cpu")
            )
    else:
        distributed.run_rank_zero(
            callback, use_ddp=True, device=torch.device("cpu")
        )
    assert events == (
        ["barrier", "callback", "broadcast"]
        if rank == 0
        else ["barrier", "broadcast"]
    )


def test_non_ddp_callback_uses_no_collectives(monkeypatch):
    monkeypatch.setattr(
        distributed.dist,
        "barrier",
        Mock(side_effect=AssertionError("collective")),
    )
    monkeypatch.setattr(
        distributed.dist,
        "broadcast",
        Mock(side_effect=AssertionError("collective")),
    )
    callback = Mock()
    distributed.run_rank_zero(
        callback, use_ddp=False, device=torch.device("cpu")
    )
    callback.assert_called_once_with()


@pytest.mark.parametrize("use_ema", [False, True])
def test_best_checkpoint_has_portable_model_keys(tmp_path, use_ema):
    from rebm.training.misc import save_model

    base = nn.Linear(2, 2)
    source = nn.DataParallel(AveragedModel(base) if use_ema else base)
    path = tmp_path / "best.pth"
    save_model(unwrap_model(source), str(path))
    weights = torch.load(path, weights_only=True)
    restored = nn.Linear(2, 2)
    restored.load_state_dict(weights, strict=True)
    assert set(weights) == {"weight", "bias"}
    assert torch.equal(restored.weight, base.weight)


def test_training_state_publication_is_atomic(
    monkeypatch, tmp_path, cfg, training
):
    modeling = training.rebm.training.modeling
    cfg.wandb_dir = str(tmp_path)
    monkeypatch.setattr(training.wandb, "run", SimpleNamespace(id="test-run"))
    model = nn.DataParallel(nn.Linear(2, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = Path(
        modeling.get_training_state_path(
            str(tmp_path), cfg.wandb_project, "test-run"
        )
    )
    path.parent.mkdir(parents=True)
    torch.save({"previous": True}, path)
    original_save = torch.save

    def observe_save(state, target):
        assert str(target) == str(path) + ".tmp"
        assert torch.load(path, weights_only=True) == {"previous": True}
        original_save(state, target)

    monkeypatch.setattr(torch, "save", observe_save)
    modeling.save_training_state(
        model,
        optimizer,
        2,
        1,
        training.ImageGenerationMetrics(),
        training.ClassificationMetrics(),
        cfg,
        wandb_run_id="test-run",
    )
    state = torch.load(path, weights_only=False)
    assert state["training_state"]["global_step"] == 2
    assert set(state["model_state_dict"]) == {"module.weight", "module.bias"}
    assert not Path(str(path) + ".tmp").exists()


@pytest.mark.parametrize("use_ema", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_restores_modes_and_grad_flags(monkeypatch, use_ema, fail):
    base = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    base.train()
    base[1].eval()  # Preserve mixed train/eval modes, not only the root flag.
    base[0].bias.requires_grad_(False)
    model = nn.DataParallel(base)
    ema = AveragedModel(base) if use_ema else None
    source = ema if use_ema else model
    modes = [m.training for m in source.modules()]
    flags = [p.requires_grad for p in source.parameters()]
    parameters = [p.detach().clone() for p in source.parameters()]
    # CPU supports DP construction, but the DDP evaluation path must not use it.
    monkeypatch.setattr(
        nn.DataParallel,
        "__init__",
        Mock(side_effect=AssertionError("DP during DDP eval")),
    )

    def evaluate():
        with evaluation_model(model, ema, use_ddp=True) as evaluating:
            assert evaluating is unwrap_model(source)
            assert not any(m.training for m in evaluating.modules())
            assert not any(p.requires_grad for p in evaluating.parameters())
            x = torch.ones(3, 2, requires_grad=True)
            evaluating(x).sum().backward()  # Input gradients still work.
            assert x.grad is not None
            assert all(p.grad is None for p in evaluating.parameters())
            if fail:
                raise ValueError("evaluation failed")

    if fail:
        with pytest.raises(ValueError, match="evaluation failed"):
            evaluate()
    else:
        evaluate()
    assert [m.training for m in source.modules()] == modes
    assert [p.requires_grad for p in source.parameters()] == flags
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters, source.parameters())
    )
    assert all(p.grad is None for p in source.parameters())


@pytest.mark.parametrize("use_ema", [False, True])
def test_non_ddp_evaluation_keeps_data_parallel(use_ema):
    model = nn.DataParallel(nn.Linear(2, 2))
    ema = AveragedModel(model.module) if use_ema else None
    with evaluation_model(model, ema, use_ddp=False) as evaluating:
        assert isinstance(evaluating, nn.DataParallel)
        assert unwrap_model(evaluating) is unwrap_model(
            ema if use_ema else model
        )


@pytest.mark.parametrize(
    "rank,enabled,num_classes",
    [(0, True, 2), (1, True, 2), (0, False, 2), (0, True, 1)],
)
def test_validation_loader_is_full_and_rank_zero_only(
    monkeypatch, cfg, training, rank, enabled, num_classes
):
    cfg.n_imgs_per_evaluation_log = 16 if enabled else None
    cfg.data.num_classes = num_classes
    calls = []
    dataset = torch.utils.data.TensorDataset(torch.arange(7), torch.arange(7))

    def loader(**kwargs):
        calls.append(kwargs)
        sampler = None
        if kwargs.get("use_ddp", False):
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=2, rank=rank
            )
        return torch.utils.data.DataLoader(
            dataset, batch_size=kwargs["batch_size"], sampler=sampler
        )

    monkeypatch.setattr(
        training.rebm.training.data, "get_indist_dataloader", loader
    )
    monkeypatch.setattr(
        training.rebm.training.data, "get_outdist_dataloader", loader
    )
    loaders = training.create_dataloaders(cfg, rank, 2)
    validation = [c for c in calls if c.get("split") == "val"]
    if rank == 0 and enabled and num_classes > 1:
        assert len(validation) == 1
        assert validation[0]["use_ddp"] is False
        assert validation[0]["batch_size"] == 4
        assert torch.cat(
            [batch[0] for batch in loaders.test_loader]
        ).tolist() == list(range(7))
    else:
        assert not validation
        assert loaders.test_loader is None
    assert all(c["use_ddp"] for c in calls if c.get("split") != "val")


@pytest.mark.parametrize(
    "use_ddp,use_ema",
    [(True, False), (True, True), (False, False), (False, True)],
)
@pytest.mark.parametrize(
    "log_fid,num_classes", [(True, 2), (False, 2), (True, 1), (False, 1)]
)
def test_evaluation_pipeline_and_checkpoint_weights(
    monkeypatch, cfg, training, use_ddp, use_ema, log_fid, num_classes
):
    cfg.use_ddp, cfg.use_ema = use_ddp, use_ema
    cfg.image_log.log_fid = log_fid
    cfg.data.num_classes = num_classes
    cfg.robust_eval = True
    world_size = 2 if use_ddp else 1
    model = nn.DataParallel(nn.Linear(2, 2))
    ema = AveragedModel(model.module) if use_ema else None
    expected = unwrap_model(ema if use_ema else model)
    fid_eval = Mock(return_value=(12.5, 3))
    acc_eval = Mock(return_value=0.8)
    robust_eval = Mock(return_value=0.6)
    generate = Mock(return_value=torch.zeros(2, 3, 4, 4))
    saved_fid, saved_acc, logged, grid = Mock(), Mock(), Mock(), Mock()
    monkeypatch.setattr(training, "evaluate_image_generation", fid_eval)
    monkeypatch.setattr(training, "eval_acc", acc_eval)
    monkeypatch.setattr(training, "eval_robust_acc", robust_eval)
    monkeypatch.setattr(training, "log_generate_images", generate)
    monkeypatch.setattr(training, "log_image_grid", grid)
    monkeypatch.setattr(training.wandb, "log", logged)
    monkeypatch.setattr(
        training.rebm.training.modeling, "save_best_fid_model", saved_fid
    )
    monkeypatch.setattr(
        training.rebm.training.modeling, "save_best_accuracy_model", saved_acc
    )
    loader = object()
    fid_metrics = training.ImageGenerationMetrics()
    acc_metrics = training.ClassificationMetrics()
    training.evaluate_training_model(
        model, ema, cfg, world_size, loader, fid_metrics, acc_metrics, 3
    )
    eval_cfg = generate.call_args.kwargs["cfg"]
    assert eval_cfg.batch_size == 8 // world_size
    assert cfg.batch_size == 8
    assert generate.call_args.kwargs["samples"] <= eval_cfg.batch_size
    assert fid_eval.call_count == int(log_fid)
    assert acc_eval.call_count == robust_eval.call_count == int(num_classes > 1)
    if log_fid:
        assert fid_eval.call_args.args[1] is eval_cfg
        saved_fid.assert_called_once_with(expected)
        assert fid_metrics.best_fid == 12.5
    if num_classes > 1:
        assert acc_eval.call_args.kwargs["dataloader"] is loader
        assert acc_eval.call_args.kwargs["device"] == torch.device("cpu")
        saved_acc.assert_called_once_with(expected)
    assert all(call.kwargs["step"] == 24 for call in logged.call_args_list)
    assert grid.call_args.args[-1] == 24
    assert model.training


@pytest.mark.parametrize("rank", [0, 1])
def test_disabled_evaluation_and_rank_dispatch(
    monkeypatch, cfg, training, rank
):
    # Exercise train's scheduling/dispatch itself, including first-step eval.
    # Losses and I/O are substitutes; the control flow under test is real.
    cfg.total_epochs = 1
    cfg.attack.start_step = cfg.attack.max_steps = 1
    cfg.use_ema = False
    cfg.n_imgs_per_evaluation_log = None
    model = nn.DataParallel(nn.Linear(2, 2))
    batch = (torch.ones(2, 2), torch.zeros(2, dtype=torch.long))
    loader = [batch, batch]
    data = training.Dataloaders(
        loader,
        loader,
        training.infinite_iter(loader),
        training.infinite_iter(loader),
        training.infinite_iter(loader),
        None,
    )
    monkeypatch.setattr(
        training,
        "setup_distributed",
        lambda cfg: (rank, 2, torch.device("cpu")),
    )
    monkeypatch.setattr(training, "create_dataloaders", lambda *a, **kw: data)
    monkeypatch.setattr(
        training.rebm.training.modeling, "get_model", lambda **kw: model
    )
    monkeypatch.setattr(
        training, "average_metric_across_ranks", lambda value, *a: value
    )
    monkeypatch.setattr(
        training, "any_rank_requests_stop", lambda *a, **kw: False
    )
    monkeypatch.setattr(training.signal, "signal", lambda *a: None)
    monkeypatch.setattr(training.wandb, "log", lambda *a, **kw: None)
    monkeypatch.setattr(training, "log_image_grid", lambda *a: None)
    monkeypatch.setattr(
        training, "compute_ebm_metrics", lambda **kw: _metrics(model, batch[0])
    )
    monkeypatch.setattr(
        training, "compute_clf_adv_loss", lambda **kw: model(batch[0]).sum()
    )
    save = Mock()
    evaluate = Mock()
    dispatch = Mock(
        side_effect=lambda callback, **kw: callback() if rank == 0 else None
    )
    monkeypatch.setattr(
        training.rebm.training.modeling, "save_training_state", save
    )
    monkeypatch.setattr(training, "evaluate_training_model", evaluate)
    monkeypatch.setattr(training, "run_rank_zero", dispatch)
    training.train(cfg)
    evaluate.assert_not_called()
    dispatch.assert_not_called()
    cfg.n_imgs_per_evaluation_log = 16
    training.train(cfg)
    assert (
        dispatch.call_count == 2
    )  # Both ranks enter, even for rank-zero work.
    assert evaluate.call_count == (2 if rank == 0 else 0)
    assert save.call_count == (4 if rank == 0 else 0)  # before/after evaluation


def _metrics(model, images):
    from rebm.training.metrics import EBMMetrics

    return EBMMetrics(
        clean_auc=0.5, adv_auc=0.5, ebm_total_loss=model(images).sum()
    )


@pytest.mark.distributed
def test_two_process_gloo_evaluation(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/ddp_evaluation_worker.py"),
            "--rendezvous",
            str(tmp_path / "store"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "two-rank evaluation passed" in result.stdout
