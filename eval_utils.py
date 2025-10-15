#!/usr/bin/env python3
"""
Shared utilities for model evaluation scripts (FID and accuracy evaluation).
"""

import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, TextIO


class TeeWriter:
    """A writer that outputs to both terminal and a log file."""

    def __init__(self, log_file_path: Path):
        """
        Initialize TeeWriter.

        Args:
            log_file_path: Path to the log file
        """
        self.log_file_path = log_file_path
        self.log_file = None

    def __enter__(self):
        """Open log file for writing."""
        self.log_file = open(self.log_file_path, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close log file."""
        if self.log_file:
            self.log_file.close()

    def write(self, data: str):
        """Write data to both terminal and log file."""
        # Write to terminal
        sys.stdout.write(data)
        sys.stdout.flush()

        # Write to log file
        if self.log_file:
            self.log_file.write(data)
            self.log_file.flush()

    def flush(self):
        """Flush both outputs."""
        sys.stdout.flush()
        if self.log_file:
            self.log_file.flush()


def load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_file)
    
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['method', 'dataset', 'checkpoint', 'model_type']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in config: {missing_fields}")
    
    return config


def sanitize_job_name(name: str) -> str:
    """Sanitize job name for SLURM."""
    return name.replace(' ', '_').replace(':', '_')


def check_job_completed(log_file: Path, completion_marker: str) -> bool:
    """
    Check if job is already completed by looking for completion marker in log file.
    
    Args:
        log_file: Path to log file
        completion_marker: String to search for (e.g., 'FID:', 'robust accuracy:')
    
    Returns:
        True if completion marker found, False otherwise
    """
    if not log_file.exists():
        return False
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            return completion_marker.lower() in content.lower()
    except IOError:
        return False



def create_log_directory(dataset: str, eval_type: str) -> Path:
    """
    Create and return log directory path based on dataset and evaluation type.

    Args:
        dataset: Dataset name (cifar10, cifar100, restrictedimagenet, imagenet)
        eval_type: Evaluation type ('fid', 'acc', 'ood_detection', or 'train')

    Returns:
        Path to log directory
    """
    if eval_type == 'fid':
        log_dir = Path(f"slurm_log/eval_fid/{dataset}")
    elif eval_type == 'acc':
        if dataset in ["imagenet", "restrictedimagenet"]:
            log_dir = Path(f"slurm_log/eval_acc/imagenet")
        else:
            log_dir = Path(f"slurm_log/eval_acc")
    elif eval_type == 'ood_detection':
        log_dir = Path(f"slurm_log/ood_detection/{dataset}")
    elif eval_type == 'train':
        log_dir = Path(f"slurm_log/train/{dataset}")
    else:
        raise ValueError(f"Unsupported eval_type: {eval_type}")

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir



def get_available_partition() -> str:
    """Return first available partition that is up and has idle nodes."""
    try:
        for partition in ['mi3258x', 'mi3008x', 'mi2508x', 'mi2104x']:
            result = subprocess.run(['sinfo', '-p', partition],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                partition_up = any('up' in line for line in lines)
                has_idle = any('idle' in line for line in lines)
                if partition_up and has_idle:
                    return partition
        return 'mi2508x'  # Fallback
    except:
        return 'mi2508x'  # Fallback


def get_partition_time_limit(partition: str) -> str:
    """Get the maximum time limit for a SLURM partition."""
    time_limits = {'mi3008x': '24:00:00', 'mi3258x': '12:00:00', 'mi2508x': '48:00:00', 'mi2104x': '24:00:00', 'devel': '2:00:00', 'mi2101x': '48:00:00'}
    return time_limits.get(partition, '24:00:00')
