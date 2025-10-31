#!/usr/bin/env python3
"""
Training Script - Launch training jobs from YAML model configuration files
Reads configuration files and launches training using the train_cmd field.
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch training jobs using YAML model configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml --partition mi3008x
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
            "mi2508x",
            "mi2104x",
            "devel",
            "mi2101x",
            "local",
        ],
        default="local",
        help='SLURM partition to use or "local" for local execution (default: local)',
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )

    parser.add_argument(
        "--requeue",
        action="store_true",
        default=False,
        help="Enable SLURM --requeue for automatic resumption after timeout/failure",
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


def submit_job(
    job_name: str,
    log_file: Path,
    train_cmd: str,
    wandb_name: str = None,
    partition: str = None,
    python_bin: str = "python",
    requeue: bool = False,
) -> bool:
    """Submit a job (SLURM if available, otherwise local) and return True if successful."""

    train_cmd = train_cmd.strip()
    if train_cmd.startswith("-m "):
        train_cmd_parts = [python_bin] + shlex.split(train_cmd)
    else:
        train_cmd_parts = shlex.split(train_cmd)

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

        # Build environment variables for the command
        env_vars = []
        if wandb_name:
            env_vars.append(f"WANDB_NAME={wandb_name}")

        # Generate WANDB_RUN_ID if requeue is enabled
        # This allows automatic state file discovery after requeue
        if requeue:
            import wandb
            wandb_run_id = wandb.util.generate_id()
            env_vars.append(f"WANDB_RUN_ID={wandb_run_id}")

        train_cmd_str = " ".join(env_vars + train_cmd_parts)

        # Use SLURM's %j placeholder for job ID in log filename
        log_file_with_jobid = str(log_file).replace(".log", "-%j.log")

        sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--output={log_file_with_jobid}",
            "--nodes=1",
            "--ntasks=1",
            f"--time={time_limit}",
            f"--partition={partition}",
            "--signal=SIGUSR1@300",  # Send SIGUSR1 300 seconds (5 min) before timeout
            f"--wrap={train_cmd_str}",
        ]

        # Add --requeue flag if enabled
        if requeue:
            sbatch_cmd.insert(7, "--requeue")  # Insert after --partition

        try:
            subprocess.run(
                sbatch_cmd, capture_output=True, text=True, check=True
            )
            print(
                f"Successfully submitted SLURM job: {job_name} on partition {partition}"
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job {job_name}: {e}")
            print(f"Error output: {e.stderr}")
            return False
    else:
        try:
            print("Running training locally")

            # Set WANDB_NAME environment variable if provided
            env = os.environ.copy()
            if wandb_name:
                env['WANDB_NAME'] = wandb_name

            with TeeWriter(log_file) as tee:
                process = subprocess.Popen(
                    train_cmd_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
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
    dry_run: bool = False,
    verbose: bool = True,
    requeue: bool = False,
) -> bool:
    """
    Run training using YAML configuration file.

    Args:
        config_file: Path to YAML model configuration file
        partition: SLURM partition to use (if None, uses automatic detection)
        dry_run: If True, print command without executing
        verbose: Whether to print progress messages
        requeue: If True, enable SLURM --requeue for automatic resumption

    Returns:
        True if successful, False otherwise
    """
    try:
        config = load_train_yaml_config(config_file)
    except (FileNotFoundError, ValueError) as e:
        if verbose:
            print(f"Error loading config: {e}")
        raise

    if verbose:
        print(f"Loaded config: {config_file}")
        print(f"Dataset: {config['dataset']}")
        print(f"Method: {config['method']}")
        print(f"Model type: {config['model_type']}")
        print(f"Train command: {config['train_cmd']}")

    if dry_run:
        train_cmd = config["train_cmd"].strip()
        if train_cmd.startswith("-m "):
            full_cmd = f"python {train_cmd}"
        else:
            full_cmd = train_cmd
        print(f"Would execute: {full_cmd}")
        return True

    dataset = config["dataset"]
    log_dir = create_log_directory(dataset, "train")

    config_name = Path(config_file).stem
    job_name = f"train_{config_name}"
    safe_job_name = sanitize_job_name(job_name)

    log_file = log_dir / f"{config_name}.log"

    print(f"Log file: {log_file}")

    if verbose:
        print(f"Submitting training job: {config_name}")
        print(f"WandB run name: {config_name}")

    if submit_job(safe_job_name, log_file, config["train_cmd"], wandb_name=config_name, partition=partition, requeue=requeue):
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
            dry_run=args.dry_run,
            verbose=True,
            requeue=args.requeue,
        )
        if not success:
            sys.exit(1)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
