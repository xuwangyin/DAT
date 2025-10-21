# Distributed Data Parallel (DDP) Implementation

## Overview
Phase 1: Basic DDP support has been implemented for the training codebase. The implementation allows training to scale across multiple GPUs using PyTorch's DistributedDataParallel.

## Changes Made

### 1. Configuration (rebm/training/config_classes.py)
- Added `use_ddp: bool = True` - Enable/disable DDP training (default: True)
- Added `ddp_backend: str = "nccl"` - Backend for distributed training

### 2. Distributed Initialization (rebm/training/train.py)
- Added `setup_distributed()` function that:
  - Initializes process group when DDP is enabled
  - Returns rank, world_size, and device for each process
  - Warns when DDP is used with only 1 GPU
  - Falls back to DataParallel mode when `use_ddp=False`

### 3. Model Wrapper (rebm/training/modeling.py)
- Updated `get_model()` to support both DataParallel and DDP:
  - Added `use_ddp` and `rank` parameters
  - Wraps model with DDP when enabled: `DDP(model, device_ids=[rank])`
  - Falls back to DataParallel otherwise
- Updated checkpoint loading to handle both DataParallel and DDP state dicts

### 4. Data Loading (rebm/training/data.py)
- Updated `get_indist_dataloader()` and `get_outdist_dataloader()` to support DistributedSampler
- Added `use_ddp`, `rank`, and `world_size` parameters
- When DDP is enabled, uses `DistributedSampler` for data sharding across GPUs
- Sampler handles shuffling when DDP is active

### 5. Rank Guards (rebm/training/train.py)
- Added rank checks to ensure only rank 0 performs logging and checkpointing:
  - `wandb.log()` calls guarded with `if rank == 0`
  - Checkpoint saving guarded with `and rank == 0`
  - Evaluation functions return early if `rank != 0`
  - Image logging only happens on rank 0

## Usage

### Single GPU (DataParallel)
```bash
python -m rebm.training.train experiments/cifar10/config.yaml use_ddp=False
```

### Multi-GPU with DDP (Recommended)
```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 -m rebm.training.train experiments/cifar10/config.yaml

# Or with explicit environment variables
RANK=0 WORLD_SIZE=4 MASTER_ADDR=localhost MASTER_PORT=12355 \
python -m rebm.training.train experiments/cifar10/config.yaml
```

### Environment Variables for DDP
When using torchrun, these are set automatically:
- `RANK`: Process rank (0 to world_size-1)
- `WORLD_SIZE`: Total number of processes
- `MASTER_ADDR`: Address of rank 0 process
- `MASTER_PORT`: Port for communication

## Key Design Decisions

### 1. DDP as Default
- `use_ddp=True` by default to encourage multi-GPU usage
- Can be disabled with `use_ddp=False` for single GPU or debugging

### 2. Per-GPU Batch Size
- `batch_size` in config is per-GPU batch size
- Effective batch size = `batch_size * world_size`
- Example: `batch_size=128` with 4 GPUs = effective batch size of 512

### 3. RNG Seeding
- Each rank gets different seed: `seed + rank`
- Ensures data augmentation diversity across GPUs
- Maintains reproducibility within each rank

### 4. EMA Model Handling
- EMA model is NOT wrapped in DDP
- Only the base model is wrapped
- EMA updates happen independently on each GPU

### 5. Evaluation Strategy
- Only rank 0 performs evaluation
- Saves computation on other ranks
- Evaluation uses rank 0's model state

## Implementation Notes

### What's Synchronized
- Model gradients (automatic via DDP)
- Model parameters (automatic via DDP)
- Optimizer steps (each rank independently, but on synchronized params)

### What's NOT Synchronized
- Dataloader iteration (each rank sees different data)
- Random number generation (intentionally different per rank)
- Logging and checkpointing (only rank 0)
- EMA model updates (independent per rank, but should converge due to synced base model)

### Checkpoint Compatibility
- Checkpoints saved by DDP have `module.` prefix
- `load_checkpoint()` automatically handles prefix mismatches
- Can load DDP checkpoints into DataParallel and vice versa

## Testing

### Verify DDP is Working
1. Check logs for "Initialized DDP with N processes"
2. Monitor GPU utilization across all GPUs
3. Check that effective batch size matches expectations
4. Verify training speed improvement scales with GPU count

### Common Issues
1. **NCCL initialization hanging**: Check network/firewall settings
2. **OOM errors**: Reduce per-GPU batch size
3. **Different results per run**: Expected due to non-determinism in DDP
4. **Slow training**: Check for CPU bottlenecks in data loading

## Performance Expectations

### Speedup (vs Single GPU)
- 2 GPUs: ~1.8x speedup
- 4 GPUs: ~3.5x speedup
- 8 GPUs: ~7x speedup

*Note: Actual speedup depends on model size, batch size, and communication overhead*

### When NOT to Use DDP
- Single GPU (overhead without benefit)
- Small models (communication overhead dominates)
- Debugging (harder to trace issues)

## Future Enhancements (Not in Phase 1)

### Training Resumption
- Save/load full training state including:
  - Global step counter
  - Epoch counter
  - Attack step progression
  - EMA model state
  - Metrics history
- See `TRAINING_RESUMPTION_PLAN.md` for details

### Advanced DDP Features
- Gradient accumulation with DDP
- Mixed precision training (AMP) with DDP
- Zero Redundancy Optimizer (ZeRO)
- Model parallelism for very large models

### Monitoring & Debugging
- Per-GPU metrics logging
- Gradient synchronization monitoring
- Communication overhead profiling

## Files Modified

1. `rebm/training/config_classes.py` - Added DDP config fields
2. `rebm/training/train.py` - Added distributed setup and rank guards
3. `rebm/training/modeling.py` - Added DDP model wrapping
4. `rebm/training/data.py` - Added DistributedSampler support

## References

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- Existing DDP examples in `InNOutRobustnessMean0_cifar100/`
