#!/usr/bin/env python3
"""
FID Evaluation Script - Python Version
Converts the bash script for running FID evaluations on various datasets and models.
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
        description="Run FID evaluation using YAML model configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml --steps 20 --batch-size 500
        """
    )
    
    parser.add_argument(
        'config_file',
        help='Path to YAML model configuration file'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        help='Override number of steps for evaluation'
    )
    
    parser.add_argument(
        '-bs', '--batch-size',
        type=int,
        default=1000,
        help='Batch size for evaluation (default: 1000)'
    )
    
    parser.add_argument(
        '--unconditional',
        action='store_true',
        help='Enable unconditional generation using logsumexp sampling'
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


def load_fid_yaml_config(config_file: str) -> dict:
    """Load YAML configuration file with FID-specific validation."""
    config = load_yaml_config(config_file)
    
    # Validate FID-specific required fields
    fid_required_fields = ['fid_eval_config', 'fid_num_steps']
    missing_fields = [field for field in fid_required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing FID-specific required fields in config: {missing_fields}")
    
    return config




def check_fid_job_completed(log_file: Path) -> bool:
    """Check if FID job is already completed by looking for 'FID:' in log file."""
    return check_job_completed(log_file, 'FID:')


def submit_job(
    job_name: str,
    log_file: Path,
    template_config: str,
    ckpt_path: Path,
    num_steps: int,
    model_type: str,
    batch_size: int = 1000,
    unconditional: bool = False,
    partition: str = None,
    python_bin: str = "python"
) -> bool:
    """Submit a job (SLURM if available, otherwise local) and return True if successful."""
    
    # Construct the command with new positional config + KEY=VALUE format
    python_cmd_parts = [
        python_bin, "-m", "rebm.eval.eval_fid",
        template_config,  # Positional config file argument
        f"model.ckpt_path={ckpt_path}",
        f"image_log.num_samples=50000",
        f"image_log.num_steps={num_steps}",
        f"batch_size={batch_size}",
        f"model.model_type={model_type}"
    ]

    # Add unconditional generation flag if specified
    if unconditional:
        python_cmd_parts.append("logsumexp_sampling=True")

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
        time_limits = {'mi3258x': '4:00:00', 'mi3008x': '4:00:00', 'mi2508x': '12:00:00', 'mi2104x': '24:00:00', 'devel': '00:30:00', 'mi2101x': '12:00:00'}
        time_limit = time_limits.get(partition, '4:00:00')

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
                print(f"Successfully completed FID evaluation: {ckpt_path}")
                return True
            else:
                print(f"FID evaluation failed with return code {process.returncode}: {ckpt_path}")
                return False
        except Exception as e:
            print(f"Error running FID evaluation {ckpt_path}: {e}")
            return False


def run_fid_evaluation_from_config(
    config_file: str,
    steps_override: int = None,
    batch_size: int = 1000,
    unconditional: bool = False,
    partition: str = None,
    skip_completed: bool = False,
    verbose: bool = True
) -> tuple[int, int, int]:
    """
    Run FID evaluation using YAML configuration file.

    Args:
        config_file: Path to YAML model configuration file
        steps_override: Optional steps override
        batch_size: Batch size for evaluation (default: 1000)
        unconditional: Enable unconditional generation using logsumexp sampling
        partition: SLURM partition to use (if None, uses automatic detection)
        skip_completed: Whether to skip if job already completed (default: False)
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (submitted_count, skipped_count, total_jobs)
    """
    # Load configuration
    try:
        config = load_fid_yaml_config(config_file)
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
    
    # Create log directory
    dataset = config['dataset']
    log_dir = create_log_directory(dataset, 'fid')
    
    # Use override steps if provided, otherwise use config
    num_steps = steps_override if steps_override is not None else config['fid_num_steps']
    
    # Extract checkpoint info
    ckpt_path = Path(config['checkpoint'])
    template_config = config['fid_eval_config']
    model_type = config['model_type']
    
    # Generate job name and log file from config file
    config_name = Path(config_file).stem
    job_name = f"{dataset}_evalfid_{config_name}_steps{num_steps}"
    
    # Add unconditional to job name if specified
    if unconditional:
        job_name += "_unconditional"
    
    safe_job_name = sanitize_job_name(job_name)
    
    # Generate log file name with unconditional suffix if specified
    log_file_name = f"{config_name}_steps{num_steps}"
    if unconditional:
        log_file_name += "_unconditional"
    log_file_name += ".log"
    
    log_file = log_dir / log_file_name
    
    # Check if job is already completed (only if skip_completed is True)
    if skip_completed and check_fid_job_completed(log_file):
        if verbose:
            print(f"Skipping {ckpt_path}")
            print(f"Reason: Job already completed successfully (found 'FID:' in log file)")
        print(f"Log file: {log_file}")
        return 0, 1, 1
    
    # Output log file before submitting job
    print(f"Log file: {log_file}")

    # Submit the job
    if verbose:
        print(f"Submitting job for checkpoint: {ckpt_path} with {num_steps} steps")

    if submit_job(
        safe_job_name,
        log_file,
        template_config,
        ckpt_path,
        num_steps,
        model_type,
        batch_size,
        unconditional,
        partition
    ):
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
        run_fid_evaluation_from_config(
            config_file=args.config_file,
            steps_override=args.steps,
            batch_size=args.batch_size,
            unconditional=args.unconditional,
            partition=args.partition,
            skip_completed=args.skip_completed,
            verbose=True
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
