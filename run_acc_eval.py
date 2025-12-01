#!/usr/bin/env python3
"""
Accuracy Evaluation Script - Python Version
Converts the bash script for running robustness accuracy evaluations on various datasets and models.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Import shared utilities
from eval_utils import (
    TeeWriter,
    check_job_completed,
    create_log_directory,
    get_available_partition,
    get_partition_time_limit,
    load_yaml_config,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run robustness accuracy evaluation using YAML model configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model_configs/cifar10-dat-WideResNet34x10-T40-seed0.yaml
  %(prog)s model_configs/imagenet-dat-ResNet50ImageNet-T15-300K-seed0.yaml --batch-size 100
        """,
    )

    parser.add_argument(
        "config_file", help="Path to YAML model configuration file"
    )

    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for evaluation (default: 200)",
    )

    parser.add_argument(
        "-t",
        "--threat-model",
        type=str,
        choices=["L2", "Linf"],
        help="Override threat model from config (L2 or Linf)",
    )

    parser.add_argument(
        "--eps",
        type=float,
        help="Override epsilon from config",
    )

    parser.add_argument(
        "--calibration",
        action="store_true",
        default=False,
        help="Generate calibration analysis instead of robustness evaluation",
    )

    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        choices=["mi3258x", "mi3008x", "mi2508x", "mi2104x", "local"],
        default="local",
        help='SLURM partition to use or "local" for local execution (default: local)',
    )

    return parser.parse_args()


def load_acc_yaml_config(config_file: str) -> dict:
    """Load YAML configuration file with accuracy-specific validation."""
    config = load_yaml_config(config_file)

    # Validate accuracy-specific required fields
    acc_required_fields = ["pgd_epsilon"]
    missing_fields = [
        field for field in acc_required_fields if field not in config
    ]
    if missing_fields:
        raise ValueError(
            f"Missing accuracy-specific required fields in config: {missing_fields}"
        )

    return config


def check_acc_job_completed(log_file: Path) -> bool:
    """Check if accuracy job is already completed by looking for 'robust accuracy:' in log file."""
    return check_job_completed(log_file, "robust accuracy:")


def submit_job(
    job_name: str,
    log_file: Path,
    checkpoint: str,
    dataset: str,
    model_type: str,
    pgd_epsilon: float,
    batch_size: int = 200,
    python_bin: str = "python",
    calibration: bool = False,
    calibration_save_path: str = None,
    partition: str = None,
    config: dict = None,
) -> bool:
    """Submit a job (SLURM if available, otherwise local) and return True if successful."""

    if "cifar" in dataset.lower():
        # CIFAR dataset evaluation
        script_path = (
            "InNOutRobustnessMean0_cifar100/cifar10_robustness_test.py"
        )

        # Map ImageNet model types for cifar10_robustness_test.py
        model_type_arg = {
            "ResNet50ImageNet": "resnet50",
            "WideResNet50x4ImageNet": "wide_resnet50_4",
        }.get(model_type, model_type)

        python_cmd = [
            python_bin,
            "-u",
            script_path,
            "--model_type",
            model_type_arg,
            "--checkpoint",
            checkpoint,
            "--dataset",
            dataset,
            "--distance_type",
            "L2",
            "--eps",
            str(pgd_epsilon),
            "--bs",
            str(batch_size),
        ]

        # Add calibration arguments if needed
        if calibration:
            python_cmd.extend(
                [
                    "--calibration_save_path",
                    calibration_save_path,
                    "--calibration_only",
                ]
            )

    elif dataset.lower() == "imagenet":
        # ImageNet dataset evaluation - unified model type conversion
        model_type_mapping = {
            "ResNet50ImageNet": "resnet50",
            "WideResNet50x4ImageNet": "wide_resnet50_4",
            "convnext_large": "convnext_large",
        }

        if model_type not in model_type_mapping:
            raise ValueError(f"Unsupported ImageNet model type: {model_type}")

        converted_model_type = model_type_mapping[model_type]

        if calibration:
            # ImageNet calibration - use CIFAR script
            script_path = (
                "InNOutRobustnessMean0_cifar100/cifar10_robustness_test.py"
            )

            python_cmd = [
                python_bin,
                "-u",
                script_path,
                "--model_type",
                converted_model_type,
                "--checkpoint",
                checkpoint,
                "--dataset",
                dataset,
                "--distance_type",
                "L2",
                "--eps",
                str(pgd_epsilon),
                "--bs",
                str(batch_size),
                "--calibration_save_path",
                calibration_save_path,
                "--calibration_only",
            ]
        else:
            # ImageNet robustness evaluation
            script_path = "evaluate_imagenet_robustbench.py"

            # Get threat model from config, default to L2
            threat_model = config.get("threat_model", "L2")

            python_cmd = [
                python_bin,
                "-u",
                script_path,
                "--checkpoint",
                checkpoint,
                "--data_dir",
                "./data/ImageNet",
                "--threat_model",
                threat_model,
                "--eps",
                str(pgd_epsilon),
                "--n_examples",
                "5000",  # Maximum supported by RobustBench
                "--batch_size",
                str(batch_size),
                "--architecture",
                converted_model_type,
            ]

            # Add image size if specified in config, otherwise defaults to 224
            if "image_size" in config:
                python_cmd.extend(["--img_size", str(config["image_size"])])
    else:
        # Catch-all for unsupported datasets
        raise ValueError(
            f"Unsupported dataset: {dataset}. Supported datasets: CIFAR variants, ImageNet"
        )

    # Check if slurm is available and not forced to use local
    if partition == "local":
        use_slurm = False
    else:
        try:
            subprocess.run(["which", "sbatch"], capture_output=True, check=True)
            use_slurm = True
        except subprocess.CalledProcessError:
            use_slurm = False

    if use_slurm:
        # SLURM submission
        if partition is None:
            partition = get_available_partition()
        time_limit = get_partition_time_limit(partition)

        sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--output={log_file}",
            "--nodes=1",
            "--ntasks=1",
            f"--time={time_limit}",
            f"--partition={partition}",
        ]

        # Add email/SMS notifications if configured
        mail_user = os.environ.get("SLURM_MAIL_USER")
        if mail_user:
            sbatch_cmd.extend([
                f"--mail-user={mail_user}",
                "--mail-type=BEGIN,END,FAIL,TIME_LIMIT",
            ])

        sbatch_cmd.append(f"--wrap={' '.join(python_cmd)}")

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
        # Local execution with streaming output to both terminal and log file
        try:
            print(f"Running locally: {checkpoint}")
            with TeeWriter(log_file) as tee:
                process = subprocess.Popen(
                    python_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                # Stream output line by line
                for line in process.stdout:
                    tee.write(line)

                process.wait()

            if process.returncode == 0:
                print(f"Successfully completed evaluation: {checkpoint}")
                return True
            else:
                print(
                    f"Evaluation failed with return code {process.returncode}: {checkpoint}"
                )
                return False
        except Exception as e:
            print(f"Error running evaluation {checkpoint}: {e}")
            return False


def run_acc_evaluation_from_config(
    config_file: str,
    batch_size: int = 200,
    verbose: bool = True,
    calibration: bool = False,
    partition: str = None,
    threat_model: str = None,
    eps: float = None,
) -> tuple[int, int, int]:
    """
    Run accuracy evaluation using YAML configuration file.

    Args:
        config_file: Path to YAML model configuration file
        batch_size: Batch size for evaluation (default: 200)
        verbose: Whether to print progress messages
        calibration: Generate calibration analysis instead of robustness evaluation
        partition: SLURM partition to use (if None, uses automatic detection)
        threat_model: Override threat model from config (L2 or Linf)
        eps: Override epsilon from config

    Returns:
        Tuple of (submitted_count, skipped_count, total_jobs)
    """
    # Load configuration
    try:
        config = load_acc_yaml_config(config_file)
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
    dataset = config["dataset"]
    log_dir = create_log_directory(dataset, "acc")

    # Extract checkpoint info
    ckpt_path = Path(config["checkpoint"])
    model_type = config["model_type"]
    pgd_epsilon = eps if eps is not None else config["pgd_epsilon"]

    # Override threat model if specified
    if threat_model is not None:
        config["threat_model"] = threat_model

    # Generate job name and log file from config file
    config_name = Path(config_file).stem

    # Add threat model and eps to filename
    threat = config.get("threat_model", "L2")
    suffix = f"{threat}_eps{pgd_epsilon}"

    job_name = f"evalacc_{config_name}_{suffix}"
    log_file = log_dir / f"{config_name}_{suffix}.log"

    # Generate calibration save path if needed
    calibration_save_path = (
        str(log_dir / f"{config_name}_calibration.pdf") if calibration else None
    )

    # Check if job is already completed
    if check_acc_job_completed(log_file):
        if verbose:
            print(
                f"Skipping {ckpt_path} - 'robust accuracy:' already found in {log_file}"
            )
        print(f"Log file: {log_file}")
        return 0, 1, 1

    # Output log file before submitting job
    print(f"Log file: {log_file}")

    # Submit the job
    if verbose:
        print(f"Submitting job for checkpoint: {ckpt_path}")

    if submit_job(
        job_name,
        log_file,
        config["checkpoint"],
        dataset,
        model_type,
        pgd_epsilon,
        batch_size,
        calibration=calibration,
        calibration_save_path=calibration_save_path,
        partition=partition,
        config=config,
    ):
        if verbose:
            print("Job submission complete.")
            print("Submitted: 1, Skipped: 0, Total: 1")
        if calibration_save_path:
            print(
                f"Calibration diagram will be saved to: {calibration_save_path}"
            )
        return 1, 0, 1
    else:
        if verbose:
            print("Job submission failed.")
        return 0, 0, 1


def main():
    """Main function for command-line usage."""
    args = parse_arguments()

    try:
        run_acc_evaluation_from_config(
            config_file=args.config_file,
            batch_size=args.batch_size,
            verbose=True,
            calibration=args.calibration,
            partition=args.partition,
            threat_model=args.threat_model,
            eps=args.eps,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
