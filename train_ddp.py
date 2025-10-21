#!/usr/bin/env python3
"""
DDP Training Script - Launch distributed training jobs from YAML model configuration files
Uses torchrun for Distributed Data Parallel (DDP) training with automatic GPU detection.
Assumes use_ddp=True by default in the training configs.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Import shared utilities
from eval_utils import (
    TeeWriter,
    create_log_directory,
    get_available_partition,
    get_partition_time_limit,
    load_yaml_config,
    sanitize_job_name,
)


def get_partition_gpu_count(partition: str) -> int:
    """
    Get the number of GPUs available per node for a given partition.

    Based on cluster specifications:
    - devel: 1 GPU (1x MI210)
    - mi2101x: 1 GPU (1x MI210)
    - mi2104x: 4 GPUs (4x MI210)
    - mi2508x: 8 GPUs (4x MI250, each MI250 has 2 GCDs)
    - mi3001x: 1 GPU (1x MI300X)
    - mi3008x: 8 GPUs (8x MI300X)
    - mi3008x_long: 8 GPUs (8x MI300X)
    - mi3258x: 8 GPUs (8x MI325X)
    - local: Auto-detect from CUDA_VISIBLE_DEVICES or nvidia-smi/rocm-smi

    Args:
        partition: SLURM partition name or "local"

    Returns:
        Number of GPUs per node
    """
    partition_gpu_map = {
        "devel": 1,
        "mi2101x": 1,
        "mi2104x": 4,
        "mi2508x": 8,
        "mi3001x": 1,
        "mi3008x": 8,
        "mi3008x_long": 8,
        "mi3258x": 8,
    }

    if partition in partition_gpu_map:
        return partition_gpu_map[partition]
    elif partition == "local":
        # Auto-detect GPUs for local execution
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if cuda_devices:
                return len(cuda_devices.split(","))

        # Try rocm-smi (for AMD GPUs)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True,
                text=True,
                check=True,
            )
            # Count GPU lines in output
            gpu_count = len([line for line in result.stdout.split("\n") if "GPU[" in line])
            if gpu_count > 0:
                return gpu_count
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try nvidia-smi (for NVIDIA GPUs)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                check=True,
            )
            gpu_count = len(result.stdout.strip().split("\n"))
            if gpu_count > 0:
                return gpu_count
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Default to 1 GPU if detection fails
        print("Warning: Could not auto-detect GPU count, defaulting to 1")
        return 1
    else:
        # Unknown partition, default to 1
        print(f"Warning: Unknown partition '{partition}', defaulting to 1 GPU")
        return 1


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch DDP training jobs using YAML model configuration files with torchrun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml --partition mi3008x
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml --partition local --gpus 4

Notes:
  - This script automatically uses torchrun for distributed training
  - GPU count is auto-detected based on partition or can be manually specified
  - Training configs should have use_ddp=True (default)
        """,
    )

    parser.add_argument(
        "config_file", help="Path to YAML model configuration file"
    )

    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        choices=[
            "mi3258x",
            "mi3008x",
            "mi3008x_long",
            "mi2508x",
            "mi2104x",
            "devel",
            "mi2101x",
            "mi3001x",
            "local",
        ],
        default="local",
        help='SLURM partition to use or "local" for local execution (default: local)',
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (overrides auto-detection)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )

    return parser.parse_args()


def load_train_yaml_config(config_file: str) -> dict:
    """Load YAML configuration file with training-specific validation."""
    config = load_yaml_config(config_file)

    train_required_fields = ["train_cmd"]
    missing_fields = [
        field for field in train_required_fields if field not in config
    ]
    if missing_fields:
        raise ValueError(
            f"Missing training-specific required fields in config: {missing_fields}"
        )

    return config


def extract_training_config_path(train_cmd: str) -> str | None:
    """
    Extract the training config YAML path from the train_cmd.

    Example train_cmd:
    "python -m rebm.training.train experiments/cifar10/config.yaml"

    Returns the path to the training config or None if not found.
    """
    parts = shlex.split(train_cmd.strip())

    # Find the .yaml file in the command
    for part in parts:
        if part.endswith(".yaml") or part.endswith(".yml"):
            return part

    return None


def convert_train_cmd_to_torchrun(
    train_cmd: str,
    num_gpus: int,
) -> str:
    """
    Convert a standard python command to use torchrun for DDP.

    Args:
        train_cmd: Original training command (e.g., "python -m rebm.training.train config.yaml")
        num_gpus: Number of GPUs to use

    Returns:
        Modified command using torchrun

    Example:
        Input: "python -m rebm.training.train experiments/cifar10/config.yaml"
        Output: "torchrun --nproc_per_node=8 -m rebm.training.train experiments/cifar10/config.yaml"
    """
    train_cmd = train_cmd.strip()

    # Remove "python" or "python3" from the beginning if present
    train_cmd = re.sub(r"^python3?\s+", "", train_cmd)

    # Build torchrun command
    torchrun_cmd = f"torchrun --nproc_per_node={num_gpus} {train_cmd}"

    return torchrun_cmd


def submit_job(
    job_name: str,
    log_file: Path,
    train_cmd: str,
    partition: str = None,
    num_gpus: int = 1,
) -> bool:
    """
    Submit a DDP training job using torchrun.

    Args:
        job_name: Name of the job
        log_file: Path to log file
        train_cmd: Training command from config
        partition: SLURM partition or "local"
        num_gpus: Number of GPUs to use

    Returns:
        True if successful, False otherwise
    """
    # Convert train_cmd to use torchrun
    torchrun_cmd = convert_train_cmd_to_torchrun(train_cmd, num_gpus)
    train_cmd_parts = shlex.split(torchrun_cmd)

    if partition == "local":
        use_slurm = False
    else:
        try:
            subprocess.run(["which", "sbatch"], capture_output=True, check=True)
            use_slurm = True
        except subprocess.CalledProcessError:
            use_slurm = False

    if use_slurm:
        if partition is None:
            partition = get_available_partition()

        time_limit = get_partition_time_limit(partition)
        train_cmd_str = " ".join(train_cmd_parts)

        # For SLURM, we submit the job
        # Note: This cluster doesn't use GRES for GPU allocation,
        # GPUs are assigned based on the partition selection
        sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--output={log_file}",
            "--nodes=1",
            "--ntasks=1",  # Single task that spawns multiple processes via torchrun
            f"--time={time_limit}",
            f"--partition={partition}",
            f"--wrap={train_cmd_str}",
        ]

        try:
            result = subprocess.run(
                sbatch_cmd, capture_output=True, text=True, check=True
            )
            print(
                f"Successfully submitted SLURM job: {job_name} on partition {partition} with {num_gpus} GPUs"
            )
            print(f"Command: {train_cmd_str}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job {job_name}: {e}")
            print(f"Error output: {e.stderr}")
            return False
    else:
        try:
            print(f"Running DDP training locally with {num_gpus} GPUs")
            print(f"Command: {' '.join(train_cmd_parts)}")
            with TeeWriter(log_file) as tee:
                process = subprocess.Popen(
                    train_cmd_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                for line in process.stdout:
                    tee.write(line)

                process.wait()

            if process.returncode == 0:
                print("Successfully completed training")
                return True
            else:
                print(f"Training failed with return code {process.returncode}")
                return False
        except Exception as e:
            print(f"Error running training: {e}")
            return False


def run_training_from_config(
    config_file: str,
    partition: str = None,
    num_gpus: int = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Run DDP training using YAML configuration file with torchrun.

    Args:
        config_file: Path to YAML model configuration file
        partition: SLURM partition to use (if None, uses automatic detection)
        num_gpus: Number of GPUs to use (if None, auto-detected from partition)
        dry_run: If True, print command without executing
        verbose: Whether to print progress messages

    Returns:
        True if successful, False otherwise
    """
    try:
        config = load_train_yaml_config(config_file)
    except (FileNotFoundError, ValueError) as e:
        if verbose:
            print(f"Error loading config: {e}")
        raise

    # Auto-detect GPU count if not specified
    if num_gpus is None:
        num_gpus = get_partition_gpu_count(partition)

    if verbose:
        print(f"Loaded config: {config_file}")
        print(f"Dataset: {config['dataset']}")
        print(f"Method: {config['method']}")
        print(f"Model type: {config['model_type']}")
        print(f"Train command: {config['train_cmd']}")
        print(f"Using {num_gpus} GPUs with DDP")

    if dry_run:
        torchrun_cmd = convert_train_cmd_to_torchrun(
            config["train_cmd"], num_gpus
        )
        print(f"Would execute: {torchrun_cmd}")
        return True

    dataset = config["dataset"]
    log_dir = create_log_directory(dataset, "train")

    config_name = Path(config_file).stem
    job_name = f"train_{config_name}"
    safe_job_name = sanitize_job_name(job_name)

    log_file = log_dir / f"{config_name}.log"

    print(f"Log file: {log_file}")

    if verbose:
        print(f"Submitting DDP training job: {config_name}")

    if submit_job(safe_job_name, log_file, config["train_cmd"], partition, num_gpus):
        if verbose:
            print("Job submission complete.")
        return True
    else:
        if verbose:
            print("Job submission failed.")
        return False


def main():
    """Main function for command-line usage."""
    args = parse_arguments()

    try:
        success = run_training_from_config(
            config_file=args.config_file,
            partition=args.partition,
            num_gpus=args.gpus,
            dry_run=args.dry_run,
            verbose=True,
        )
        if not success:
            sys.exit(1)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
