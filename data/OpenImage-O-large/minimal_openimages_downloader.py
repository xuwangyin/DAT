#!/usr/bin/env python3
"""
Minimal script to download OpenImages dataset images by ID from a text file.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

import boto3
import botocore
from tqdm import tqdm


def read_image_ids(file_path: str) -> List[str]:
    """Read image IDs from a text file (one ID per line)."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def download_single_image(s3_client, image_id: str, section: str, output_dir: str):
    """Download a single image from OpenImages S3 bucket."""
    image_filename = f"{image_id}.jpg"
    s3_object_path = f"{section}/{image_filename}"
    local_file_path = os.path.join(output_dir, image_filename)
    
    # Skip if file already exists
    if os.path.exists(local_file_path):
        return
    
    try:
        with open(local_file_path, "wb") as dest_file:
            s3_client.download_fileobj(
                "open-images-dataset",
                s3_object_path,
                dest_file,
            )
    except Exception as e:
        print(f"Failed to download {image_id}: {e}")


def download_images(ids_file: str, output_dir: str, section: str = "train", max_workers: int = 10):
    """Download images based on IDs from a text file."""
    
    # Read image IDs
    image_ids = read_image_ids(ids_file)
    print(f"Found {len(image_ids)} image IDs to download")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up S3 client (unsigned access for public bucket)
    s3_client = boto3.client(
        's3',
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )
    
    # Download images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_single_image, s3_client, image_id, section, output_dir)
            for image_id in image_ids
        ]
        
        # Show progress
        for _ in tqdm(futures, desc="Downloading images"):
            pass


def main():
    parser = argparse.ArgumentParser(description="Download OpenImages dataset images by ID")
    parser.add_argument("ids_file", help="Text file containing image IDs (one per line)")
    parser.add_argument("output_dir", help="Directory to save downloaded images")
    parser.add_argument("--section", choices=["train", "validation", "test"], 
                       default="train", help="Dataset section (default: train)")
    parser.add_argument("--max-workers", type=int, default=10, 
                       help="Maximum number of parallel downloads (default: 10)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ids_file):
        print(f"Error: File {args.ids_file} not found")
        sys.exit(1)
    
    download_images(args.ids_file, args.output_dir, args.section, args.max_workers)
    print("Download completed!")


if __name__ == "__main__":
    main()
