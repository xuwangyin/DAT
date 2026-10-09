# DDP evaluation: proposed changes and validation

Related: [issue #3](https://github.com/xuwangyin/DAT/issues/3), still open.
These changes are a proposed correction, not confirmation that DAT training is
validated on multiple GPUs or nodes. Continue using offline evaluation for
production DDP runs until the hardware checks below pass.

## Evaluation behavior

- Every rank reaches the same entry barrier and completion broadcast. Rank zero
  saves training state, evaluates, then saves updated metrics before releasing
  the other ranks. Ordinary Python exceptions are communicated to every rank;
  peers do not continue to the next training forward after a failed evaluation.
- Rank zero borrows the unwrapped local model (or EMA weights). It never runs a
  DDP forward or creates DataParallel replicas on GPUs owned by other processes.
  Non-DDP training retains DataParallel evaluation.
- CUDA selection uses `LOCAL_RANK` before process-group initialization. Model
  placement, DDP `device_ids`, and the configuration's device agree. Global rank
  is used for data sharding and deciding which process logs and evaluates.
- Only rank zero creates a validation loader, without a DistributedSampler.
  It processes the full validation set, including a final partial batch.
  Disabled evaluation creates no validation loader.
- FID, adaptive FID sweeps, and sample generation use the per-rank training
  batch size; logging still counts global training images. Accuracy already
  uses the per-rank loader batch size. Sampling retains input autograd while
  model parameter gradients are disabled. Evaluation restores the previous
  per-module modes and parameter flags even on exceptions.
- Best-model saves accept plain, DP, DDP, and EMA models without depending on a
  fixed `.module.module` nesting. Training-state replacement is atomic on the
  same filesystem so offline readers do not see a partially written file.
- Shutdown requests from any rank are shared before checkpointing and exit.
  A checkpoint selected by rank-zero auto-resume is broadcast before training;
  that path must be readable by every node. The process group is destroyed on
  exit from training.

Evaluation is serialized on one GPU and can be slow. The existing 90-minute
process-group timeout still bounds how long peers can wait. A killed process,
a hung evaluator, a CUDA/NCCL fault, or evaluation longer than that timeout is
not made recoverable by exception propagation. Use offline evaluation for long
FID sweeps. This patch does not audit every DAT DDP training operation or prove
training convergence, FID correctness, or exact checkpoint-resume reproducibility.

## Safe offline workflow

1. Disable the entire in-training evaluation block with
   `n_imgs_per_evaluation_log=null`. Setting `image_log.log_fid=False` alone
   still leaves sample generation and accuracy enabled.
2. Retain separate, step-labelled checkpoints. The current trainer writes
   `training_state_latest.pth` every 20 optimization steps and replaces it;
   it does **not** automatically retain a history or save every 50k images.
   Archive completed checkpoints at a cadence appropriate for the experiment.
   Roughly 50k training images can be a starting point, not a proven optimum.
   On the old code, copy only after a save completes and while the writer is
   stopped; this patch's atomic replacement permits concurrent snapshot copies.
3. Export weights from a retained full training state, selecting EMA explicitly
   if desired. A training-state dictionary is not a model-only checkpoint:

   ```python
   import torch

   # Only load a trusted checkpoint you created.
   state = torch.load("checkpoint_step_1000.pth", map_location="cpu", weights_only=False)
   use_ema = True
   if use_ema:
       # AveragedModel stores its weights under module.* plus n_averaged.
       weights = {
           key.removeprefix("module."): value
           for key, value in state["ema_state_dict"].items()
           if key != "n_averaged"
       }
   else:
       weights = {
           key.removeprefix("module."): value
           for key, value in state["model_state_dict"].items()
       }
   torch.save(weights, "weights_step_1000.pth")
   ```

4. Use the matching architecture and evaluation data/configuration on a GPU
   not assigned to the training job, or after training finishes. For example:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m rebm.eval.eval_fid \
     experiments/cifar10/eval_fid/cifar10_conditional_wrn3410_eval_fid_steps33.yaml \
     model.ckpt_path=weights_step_1000.pth use_ddp=False batch_size=64
   ```

   The `0` above is an example GPU ordinal, **not** permission to share the
   training job's GPU 0. For accuracy, set `ckpt_path` in a matching copy of a
   `model_configs/` YAML and run `python run_acc_eval.py YOUR_MODEL_CONFIG` with
   similarly isolated devices. Keep sample counts, sampler settings, and real
   FID statistics consistent when comparing checkpoints.

## Tests without GPUs

Use Python 3.11+ and the repository's training dependencies. Initialize the
`pytorch-image-models` submodule at its pinned commit. No dataset or pretrained
checkpoint downloads and no wandb connection are used by these tests.

```bash
python -m pytest -q tests/test_ddp_evaluation.py -m 'not distributed'
```

This covers simulated global/local rank differences, CUDA binding order,
invalid local ranks, evaluation branch combinations, per-device batches versus
global logging steps, complete validation-set coverage, model-mode/gradient
restoration, checkpoint serialization, atomic checkpoint replacement, and
train-loop dispatch/disabled evaluation. FID, accuracy, dataset creation, and
wandb I/O are mocked in pipeline tests; these are not end-to-end FID tests.

The following also needs **no GPUs**, but does require local socket access:

```bash
python -m pytest -q tests/test_ddp_evaluation.py -m distributed
```

It launches two real CPU/Gloo processes and a tiny DDP model. It exercises
repeated EMA/non-EMA evaluation between optimizer steps, verifies rank-zero-only
execution and equal weights after resumed training, propagates an injected
failure, and checks a stop request originating on rank 1. The process-group
timeout is 15 seconds and the subprocess test timeout is 60 seconds.

During preparation, the isolated CPU suite was run with Python 3.12 and
PyTorch 2.14.1+cpu: **42 passed, 1 distributed test deselected**. Syntax,
whitespace, and focused lint checks also passed. The Gloo test was
**attempted but blocked before any test
collectives**: process-group creation failed with `Operation not permitted`
in Gloo's TCP device initialization in the sandbox. It has not passed here.
No CUDA devices were available.

## Validation still required on distributed hardware

First run the same small-model protocol on two available GPUs (an unused
absolute rendezvous path is required):

```bash
PYTHONPATH=. python tests/ddp_evaluation_worker.py --backend nccl \
  --rendezvous /tmp/dat-nccl-smoke-UNIQUE
```

This is a protocol smoke test only. Before declaring the DAT path fixed, also:

- Run real DAT under `torchrun` on 2 and 4 GPUs across at least two evaluation
  boundaries and subsequent optimizer steps. Exercise FID + image logging +
  clean/robust accuracy, with and without EMA. Verify only rank zero uses its
  assigned device for evaluation, with no replica allocations on peer GPUs.
- Check FID disabled, evaluation fully disabled, and single-class generation.
  Compare full validation accuracy against the standalone evaluator on the
  same checkpoint. Verify best-FID and best-accuracy files load correctly.
- Use realistic batch sizes, sample counts, and adaptive sweeps; record peak
  GPU memory and evaluation wall time relative to the 90-minute timeout.
- Inject a Python evaluation/checkpoint error and verify every rank exits;
  exercise a rank-local shutdown signal and automatic/explicit resume. Do not
  expect this to recover a fatal CUDA fault or a killed process.
- Run on at least two nodes where `RANK != LOCAL_RANK`, verifying local device
  ownership and the shared resume path. Mocked device-index tests do not
  establish multi-node correctness.

Record the commit, PyTorch/CUDA/NCCL versions, GPU topology, configuration,
commands, and logs in the PR. Keep issue #3 open until this evidence supports
closing it.
