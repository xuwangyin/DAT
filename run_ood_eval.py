#!/usr/bin/env python3
"""
OOD Detection Evaluation Script - Python Version
Converts the bash script for running OOD detection evaluations on various datasets and models.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Union

# Import shared utilities
from eval_utils import (
    load_yaml_config,
    sanitize_job_name,
    check_job_completed,
    create_log_directory,
    get_available_partition,
    TeeWriter
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OOD detection evaluation using YAML model configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml --ood-dataset noise
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml --ood-dataset svhn
        """
    )
    
    parser.add_argument(
        'config_file',
        help='Path to YAML model configuration file'
    )
    
    parser.add_argument(
        '--ood-dataset',
        type=str,
        choices=['noise', 'svhn', 'cifar100', 'cifar10', 'imagenet'],
        required=True,
        help='OOD dataset to evaluate'
    )
    
    parser.add_argument(
        '-p', '--partition',
        type=str,
        choices=['mi3258x', 'mi3008x', 'mi2508x', 'mi2104x', 'devel', 'mi2101x', 'local'],
        default='local',
        help='SLURM partition to use or "local" for local execution (default: local)'
    )

    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help='Skip evaluation if job already completed (default: always run)'
    )

    return parser.parse_args()


def load_ood_yaml_config(config_file: str) -> dict:
    """Load YAML configuration file with OOD-specific validation."""
    config = load_yaml_config(config_file)
    
    # Validate OOD-specific required fields
    # TODO: Currently using fid_eval_config for OOD detection - this is temporary
    # In the future, we should have dedicated ood_eval_config field
    ood_required_fields = ['fid_eval_config']
    missing_fields = [field for field in ood_required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing OOD-specific required fields in config: {missing_fields}")
    
    return config


def check_ood_job_completed(log_file: Path) -> bool:
    """Check if OOD detection job is already completed by looking for completion message in log file."""
    return check_job_completed(log_file, 'OOD detection evaluation completed')


def submit_job(
    job_name: str,
    log_file: Path,
    template_config: str,
    ckpt_path: Path,
    model_type: str,
    ood_dataset: str,
    partition: str = None,
    python_bin: str = "python"
) -> bool:
    """Submit a job (SLURM if available, otherwise local) and return True if successful."""

    # Construct the command with new positional config + KEY=VALUE format
    python_cmd_parts = [
        python_bin, "-m", "rebm.eval.eval_ood_detection",
        template_config,  # Positional config file argument
        f"model.ckpt_path={ckpt_path}",
        f"outdist_dataset_ood_detection={ood_dataset}",
        f"model.model_type={model_type}"
    ]

    # Check if slurm is available and not forced to use local
    if partition == 'local':
        use_slurm = False
    else:
        try:
            subprocess.run(['which', 'sbatch'], capture_output=True, check=True)
            use_slurm = True
        except subprocess.CalledProcessError:
            use_slurm = False

    if use_slurm:
        # SLURM submission
        # Determine partition to use
        if partition is None:
            partition = get_available_partition()

        # Set time limits based on partition
        time_limits = {'mi3258x': '6:00:00', 'mi3008x': '6:00:00', 'mi2508x': '12:00:00', 'mi2104x': '24:00:00', 'devel': '00:30:00', 'mi2101x': '12:00:00'}
        time_limit = time_limits.get(partition, '6:00:00')

        # SLURM sbatch command
        python_cmd_str = " ".join(python_cmd_parts)
        sbatch_cmd = [
            'sbatch',
            f'--job-name={job_name}',
            f'--output={log_file}',
            '--nodes=1',
            '--ntasks=1',
            f'--time={time_limit}',
            f'--partition={partition}',
            f'--wrap={python_cmd_str}'
        ]

        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            print(f"Successfully submitted SLURM job: {job_name} on partition {partition}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job {job_name}: {e}")
            print(f"Error output: {e.stderr}")
            return False
    else:
        # Local execution with streaming output to both terminal and log file
        try:
            print(f"Running locally: {ckpt_path}")
            with TeeWriter(log_file) as tee:
                process = subprocess.Popen(
                    python_cmd_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                # Stream output line by line
                for line in process.stdout:
                    tee.write(line)

                process.wait()

            if process.returncode == 0:
                print(f"Successfully completed OOD detection evaluation: {ckpt_path}")
                return True
            else:
                print(f"OOD detection evaluation failed with return code {process.returncode}: {ckpt_path}")
                return False
        except Exception as e:
            print(f"Error running OOD detection evaluation {ckpt_path}: {e}")
            return False


def run_ood_evaluation_from_config(
    config_file: str,
    ood_dataset: str,
    partition: str = None,
    skip_completed: bool = False,
    verbose: bool = True
) -> tuple[int, int, int]:
    """
    Run OOD detection evaluation using YAML configuration file.

    Args:
        config_file: Path to YAML model configuration file
        ood_dataset: OOD dataset to evaluate
        partition: SLURM partition to use (if None, uses automatic detection)
        skip_completed: Whether to skip if job already completed (default: False)
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (submitted_count, skipped_count, total_jobs)
    """
    
    # Load configuration
    try:
        config = load_ood_yaml_config(config_file)
    except (FileNotFoundError, ValueError) as e:
        if verbose:
            print(f"Error loading config: {e}")
        raise
    
    if verbose:
        print(f"Loaded config: {config_file}")
        print(f"Dataset: {config['dataset']}")
        print(f"Method: {config['method']}")
        print(f"Model type: {config['model_type']}")
        print(f"Checkpoint: {config['checkpoint']}")
        print(f"OOD dataset: {ood_dataset}")
    
    # Create log directory
    dataset = config['dataset']
    log_dir = create_log_directory(dataset, 'ood_detection')
    
    # Extract checkpoint info
    ckpt_path = Path(config['checkpoint'])
    # TODO: Currently using fid_eval_config for OOD detection - this is temporary
    # In the future, we should have dedicated ood_eval_config field
    template_config = config['fid_eval_config']
    model_type = config['model_type']
    
    # Generate run name from checkpoint path
    if 'wandb' in str(ckpt_path):
        # For wandb runs
        import re
        match = re.search(r'wandb/(run-\d+_\d+-[a-z0-9]+)', str(ckpt_path))
        run_name = match.group(1) if match else ckpt_path.parent.name
    else:
        # For other paths, use parent directory name
        run_name = ckpt_path.parent.name
    
    ckpt_name = ckpt_path.stem
    
    submitted_count = 0
    skipped_count = 0
    
    # Generate log file name from config file
    config_name = Path(config_file).stem
    
    # Process the OOD dataset
    job_name = f"{run_name}_{ckpt_name}_id_{dataset}_ood_{ood_dataset}"
    safe_job_name = sanitize_job_name(job_name)
    
    log_file_name = f"{config_name}_ood_{ood_dataset}.log"
    log_file = log_dir / log_file_name
    
    # Check if job is already completed (only if skip_completed is True)
    if skip_completed and check_ood_job_completed(log_file):
        if verbose:
            print(f"Skipping {ckpt_path} with OOD dataset {ood_dataset}")
            print(f"Reason: Job already completed successfully (found 'OOD detection evaluation completed' in log file)")
        print(f"Log file: {log_file}")
        return 0, 1, 1
    
    # Output log file before submitting job
    print(f"Log file: {log_file}")

    # Submit the job
    if verbose:
        print(f"Submitting OOD detection job for checkpoint: {ckpt_path}, OOD dataset: {ood_dataset}")

    if submit_job(
        safe_job_name,
        log_file,
        template_config,
        ckpt_path,
        model_type,
        ood_dataset,
        partition
    ):
        submitted_count = 1
        if verbose:
            print("Job submission complete.")
            print(f"Submitted: 1, Skipped: 0, Total: 1")
        return 1, 0, 1
    else:
        if verbose:
            print("Job submission failed.")
        return 0, 0, 1


def main():
    """Main function for command-line usage."""
    args = parse_arguments()
    
    try:
        run_ood_evaluation_from_config(
            config_file=args.config_file,
            ood_dataset=args.ood_dataset,
            partition=args.partition,
            skip_completed=args.skip_completed,
            verbose=True
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()