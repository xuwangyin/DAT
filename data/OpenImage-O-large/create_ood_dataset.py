import os
import pandas as pd
import requests
from pathlib import Path
import subprocess
from typing import List, Set
import json
from collections import defaultdict, Counter
import sys
sys.path.insert(
    0, "openimages/src"
)

def create_natural_subset_for_ood(download_dir: str = "./data/openimages", 
                                 target_images: int = 10000,
                                 min_labels_per_class: int = 10,
                                 max_labels_per_class: int = 1000):
    """
    Create a naturally diverse subset for OOD dataset creation using only training split.
    
    Args:
        download_dir: Directory to store data
        target_images: Number of images to download
        min_labels_per_class: Minimum samples per class to include
        max_labels_per_class: Maximum samples per class (for balance)
    """
    downloader = OpenImagesSubsetDownloader(download_dir)
    
    # Download metadata first (small files)
    downloader.download_metadata()
    
    # Analyze the training dataset
    analysis = downloader.analyze_label_distribution('train')
    
    # Get diverse representation using pandas-native stratified sampling
    print("Using pandas-native stratified sampling on training split...")
    selected_images = downloader.select_diverse_subset(
        target_images=target_images,
        min_labels_per_class=min_labels_per_class,
        max_labels_per_class=max_labels_per_class,
        split='train'
    )
    
    print(f"Final selection: {len(selected_images)} images from training split")
    
    
    return selected_images, downloader


class OpenImagesSubsetDownloader:
    def __init__(self, download_dir: str = "./data/openimages"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs for metadata files (training split only)
        self.urls = {
            'class_descriptions': 'https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv',
            'train_annotations': 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv'
        }
        
    def download_metadata(self):
        """Download all metadata files (small, always download these)."""
        print("Downloading metadata files...")
        
        for name, url in self.urls.items():
            # Use the actual filename from URL for consistency
            if name == 'train_annotations':
                filename = "train-annotations-human-imagelabels.csv"
            elif name == 'class_descriptions':
                filename = "class-descriptions.csv"
            else:
                filename = f"{name.replace('_', '-')}.csv"
                
            filepath = self.download_dir / filename
            
            if not filepath.exists():
                print(f"Downloading {name}...")
                response = requests.get(url)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Saved to {filepath}")
            else:
                print(f"{name} already exists, skipping")
    
    def analyze_label_distribution(self, split: str = 'train') -> pd.DataFrame:
        """Analyze distribution of labels to help with subset selection."""
        annotations_file = self.download_dir / f"{split}-annotations-human-imagelabels.csv"
        
        if not annotations_file.exists():
            self.download_metadata()
        
        print(f"Analyzing {split} label distribution...")
        df = pd.read_csv(annotations_file)
        
        # Load class descriptions
        class_desc_file = self.download_dir / "class-descriptions.csv"
        class_df = pd.read_csv(class_desc_file, header=None, names=['LabelName', 'DisplayName'])
        class_map = dict(zip(class_df['LabelName'], class_df['DisplayName']))
        
        # Count labels
        label_counts = df['LabelName'].value_counts()
        
        # Create analysis dataframe
        analysis = pd.DataFrame({
            'LabelName': label_counts.index,
            'Count': label_counts.values,
            'DisplayName': [class_map.get(label, 'Unknown') for label in label_counts.index]
        })
        
        print(f"Found {len(analysis)} unique labels across {len(df)} annotations")
        print(f"Top 10 most common labels:")
        print(analysis.head(10))
        
        return analysis
    
    def select_diverse_subset(self, 
                            target_images: int = 10000,
                            min_labels_per_class: int = 10,
                            max_labels_per_class: int = 1000,
                            split: str = 'train',
                            random_state: int = 42) -> Set[str]:
        """
        Select a diverse subset using stratified sampling.
        
        Args:
            target_images: Target number of images to download
            min_labels_per_class: Minimum samples per class
            max_labels_per_class: Maximum samples per class (for balance)
            split: Dataset split to use
            random_state: Random seed for reproducibility
            
        Returns:
            Set of image IDs to download
        """
        annotations_file = self.download_dir / f"{split}-annotations-human-imagelabels.csv"
        df = pd.read_csv(annotations_file)
        
        print(f"Performing stratified sampling from {df['LabelName'].nunique()} label classes...")
        
        # Define sampling function for each group
        def sample_group(group):
            group_size = len(group)
            if group_size < min_labels_per_class:
                # Skip groups that are too small
                return pd.DataFrame()
            
            # Determine sample size: min(group_size, max_per_class)
            sample_size = min(group_size, max_labels_per_class)
            return group.sample(n=sample_size, random_state=random_state)
        
        # Apply stratified sampling using groupby + apply
        # Suppress the deprecation warning for now
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")
            sampled_df = (df.groupby('LabelName', group_keys=False)
                           .apply(sample_group)
                           .reset_index(drop=True))
        
        print(f"Initial stratified sample: {len(sampled_df)} annotations from {sampled_df['LabelName'].nunique()} classes")
        
        # IMPORTANT: Remove duplicate images (same image can have multiple labels)
        unique_images_df = sampled_df.drop_duplicates(subset=['ImageID'])
        print(f"After removing duplicate images: {len(unique_images_df)} unique images")
        
        # If we have more images than target, do final random sampling
        if len(unique_images_df) > target_images:
            final_sample = unique_images_df.sample(n=target_images, random_state=random_state)
            print(f"Final random subsample: {len(final_sample)} images")
        else:
            final_sample = unique_images_df
            print(f"Using all {len(final_sample)} unique images")
        
        # Extract image IDs (now guaranteed to be unique)
        selected_images = set(final_sample['ImageID'].tolist())
        
        # Create dict mapping group to number of selected images
        final_image_ids = set(final_sample['ImageID'])
        final_annotations = df[df['ImageID'].isin(final_image_ids)]
        group_counts = final_annotations['LabelName'].value_counts().to_dict()
        
        # Load class descriptions to get real label names
        class_desc_file = self.download_dir / "class-descriptions.csv"
        class_df = pd.read_csv(class_desc_file, header=None, names=['LabelName', 'DisplayName'])
        class_map = dict(zip(class_df['LabelName'], class_df['DisplayName']))
        
        # Download ALL images from final_sample organized by image ID with label suffixes
        print('='*50)
        print(f'DOWNLOADING ALL {len(final_sample)} FINAL SAMPLE IMAGES BY ID')
        print('='*50)
        
        # Get the full annotations for final_sample images to preserve label associations
        final_image_ids = set(final_sample['ImageID'])
        final_annotations = df[df['ImageID'].isin(final_image_ids)]
        
        # Group by image ID to get all labels for each image
        image_to_labels = final_annotations.groupby('ImageID')['LabelName'].apply(list).to_dict()
        savedir = str(self.download_dir / split)
        
        print(f"Downloading {len(final_image_ids)} unique images with label suffixes...")
        
        # Download each image once with concatenated labels
        downloaded_count = self.download_images_by_id_with_labels(savedir, image_to_labels, class_map)
        print(f"Download complete! {downloaded_count}/{len(final_image_ids)} images downloaded to: {savedir}")
        
        # Print sampling statistics based on final unique images
        self._print_sampling_stats_unique(final_sample, sampled_df)
        
        return selected_images
    
    def _print_sampling_stats_unique(self, final_sample: pd.DataFrame, original_sample: pd.DataFrame):
        """Print statistics about unique image sampling results."""
        print(f"\nSampling Statistics:")
        print(f"Total unique images: {len(final_sample)}")
        print(f"Total annotations (before dedup): {len(original_sample)}")
        print(f"Deduplication ratio: {len(final_sample)/len(original_sample)*100:.1f}%")
        
        # Get label distribution for final unique images
        # Join back with original annotations to see label distribution
        annotations_file = self.download_dir / f"train-annotations-human-imagelabels.csv"
        df = pd.read_csv(annotations_file)
        
        final_image_ids = set(final_sample['ImageID'])
        final_annotations = df[df['ImageID'].isin(final_image_ids)]
        
        class_counts = final_annotations['LabelName'].value_counts()
        unique_classes = len(class_counts)
        
        print(f"Label classes represented: {unique_classes}")
        print(f"Annotations per class - Mean: {class_counts.mean():.1f}, "
              f"Std: {class_counts.std():.1f}")
        print(f"Annotations per class - Min: {class_counts.min()}, "
              f"Max: {class_counts.max()}")
        
        print(f"\nTop 10 most represented classes:")
        for class_name, count in class_counts.head(10).items():
            print(f"  {class_name}: {count} annotations")
        
        # Show multi-label statistics
        images_per_label_count = final_annotations.groupby('ImageID').size()
        print(f"\nMulti-label statistics:")
        print(f"Images with 1 label: {sum(images_per_label_count == 1)}")
        print(f"Images with 2+ labels: {sum(images_per_label_count > 1)}")
        print(f"Max labels per image: {images_per_label_count.max()}")
        print(f"Mean labels per image: {images_per_label_count.mean():.1f}")
    
    def download_images_by_human_labels(self, dest_dir: str, class_labels: list, label_code: str, limit: int = None):
        """
        Download images using human image labels instead of bounding box annotations.
        This allows downloading classes that have human labels but no bounding boxes.
        """
        import os
        from pathlib import Path
        import requests
        from concurrent.futures import ThreadPoolExecutor
        import logging
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger = logging.getLogger(__name__)
        
        # Read human image labels
        annotations_file = self.download_dir / "train-annotations-human-imagelabels.csv"
        df = pd.read_csv(annotations_file)
        
        # Filter for this specific label code and confidence = 1 (positive labels only)
        class_images = df[(df['LabelName'] == label_code) & (df['Confidence'] == 1)]
        
        if len(class_images) == 0:
            logger.warning(f"No positive human labels found for {label_code}")
            return
            
        # Limit the number of images if specified
        if limit is not None and len(class_images) > limit:
            class_images = class_images.sample(n=limit, random_state=42)
        
        image_ids = class_images['ImageID'].tolist()
        logger.info(f"Found {len(image_ids)} images with human labels for {class_labels[0]}")
        
        # Create destination directory
        class_name = class_labels[0].lower()
        images_dir = Path(dest_dir) / class_name / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Download images
        self._download_images_by_id_custom(image_ids, "train", str(images_dir))
        
    def _download_images_by_id_custom(self, image_ids: list, section: str, images_directory: str):
        """
        Custom function to download images by ID from OpenImages.
        """
        import os
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import logging
        
        logger = logging.getLogger(__name__)
        
        def download_single_image(image_id):
            """Download a single image."""
            # OpenImages URL pattern for images - use the AWS mirror
            url = f"https://open-images-dataset.s3.amazonaws.com/{section}/{image_id}.jpg"
            file_path = os.path.join(images_directory, f"{image_id}.jpg")
            
            # Skip if already exists
            if os.path.exists(file_path):
                return f"Skipped {image_id} (already exists)"
                
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                    
                return f"Downloaded {image_id}"
            except Exception as e:
                return f"Failed {image_id}: {str(e)}"
        
        # Download images in parallel
        logger.info(f"Downloading {len(image_ids)} images to {images_directory}")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all download tasks
            future_to_id = {executor.submit(download_single_image, img_id): img_id 
                           for img_id in image_ids}
            
            # Process completed downloads with progress bar
            successful = 0
            failed = 0
            
            for future in tqdm(as_completed(future_to_id), total=len(image_ids), desc="Downloading"):
                result = future.result()
                if "Downloaded" in result:
                    successful += 1
                elif "Skipped" in result:
                    successful += 1
                else:
                    failed += 1
                    logger.warning(result)
        
        logger.info(f"Download complete: {successful} successful, {failed} failed")

    def download_images_by_label_direct(self, dest_dir: str, display_name: str, label_code: str, image_ids: list):
        """
        Download specific images for a label, organizing them in label-specific directories.
        """
        import os
        from pathlib import Path
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import logging
        
        logger = logging.getLogger(__name__)
        
        if not image_ids:
            logger.warning(f"No image IDs provided for {display_name}")
            return
        
        # Create destination directory organized by label
        class_name = display_name.lower().replace(' ', '_').replace('/', '_')
        images_dir = Path(dest_dir) / class_name
        # images_dir.mkdir(parents=True, exist_ok=True)
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        
        def download_single_image(image_id):
            """Download a single image."""
            url = f"https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
            # file_path = os.path.join(str(images_dir), f"{image_id}.jpg")
            file_path = os.path.join(str(images_dir) + f"_{image_id}.jpg")
            
            # Skip if already exists
            if os.path.exists(file_path):
                return f"Skipped {image_id} (already exists)"
                
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                    
                return f"Downloaded {image_id}"
            except Exception as e:
                return f"Failed {image_id}: {str(e)}"
        
        # Download images in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all download tasks
            future_to_id = {executor.submit(download_single_image, img_id): img_id 
                           for img_id in image_ids}
            
            # Process completed downloads quietly
            successful = 0
            failed = 0
            
            for future in as_completed(future_to_id):
                result = future.result()
                if "Downloaded" in result:
                    successful += 1
                elif "Skipped" in result:
                    successful += 1
                else:
                    failed += 1
        
        return successful

    def load_imagenet_classes(self, imagenet_json_path: str = "./imagenet_class_index.json") -> set:
        """Load ImageNet class names for filtering."""
        import json
        
        try:
            with open(imagenet_json_path, 'r') as f:
                imagenet_data = json.load(f)
            
            imagenet_classes = set()
            for class_info in imagenet_data.values():
                common_name = class_info[1]  # e.g., "tench"
                imagenet_classes.add(common_name.lower())
                imagenet_classes.add(common_name.lower().replace('_', ' '))
                
                # # Add individual words for partial matching
                # words = common_name.lower().replace('_', ' ').split()
                # for word in words:
                #     if len(word) > 3:
                #         imagenet_classes.add(word)
            
            print(f"Loaded {len(imagenet_classes)} ImageNet class terms for filtering")
            return imagenet_classes
            
        except Exception as e:
            print(f"Warning: Could not load ImageNet classes: {e}")
            return set()

    def download_images_by_id_with_labels(self, dest_dir: str, image_to_labels: dict, class_map: dict):
        """
        Download images by ID with concatenated label suffixes.
        Each image is downloaded only once with all its labels in the filename.
        """
        import os
        from pathlib import Path
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import logging
        import re
        
        logger = logging.getLogger(__name__)
        
        # Create destination directory
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        
        def create_label_suffix(label_codes):
            """Create a clean suffix from label codes. Returns False if any label overlaps with ImageNet."""
            # Load ImageNet classes (cached)
            if not hasattr(self, '_imagenet_classes'):
                self._imagenet_classes = self.load_imagenet_classes()
            
            display_names = []
            for code in label_codes:
                display_name = class_map.get(code, code)
                display_name_lower = display_name.lower()
                
                # if (self._imagenet_classes and 
                #     (display_name_lower in self._imagenet_classes or
                #      any(word in self._imagenet_classes for word in display_name_lower.replace('_', ' ').split() if len(word) > 3))):
                #     return False  # Skip this image due to ImageNet overlap

                # Check for ImageNet overlap
                if (self._imagenet_classes and display_name_lower in self._imagenet_classes):
                    return False  # Skip this image due to ImageNet overlap
                
                # Clean the display name for filename use
                clean_name = re.sub(r'[^\w\s-]', '', display_name)  # Remove special chars
                clean_name = re.sub(r'\s+', '_', clean_name)  # Replace spaces with underscores
                clean_name = clean_name.strip('_')  # Remove leading/trailing underscores
                display_names.append(clean_name)
            
            # Sort for consistency and join with underscores
            suffix = '__'.join(sorted(display_names))
            # Truncate if too long to avoid filesystem limits
            if len(suffix) > 100:
                suffix = suffix[:100]
            return suffix
        
        def download_single_image_with_suffix(image_id, label_codes):
            """Download a single image with label suffix."""
            # Create filename with label suffix - returns False if ImageNet overlap
            label_suffix = create_label_suffix(label_codes)
            if label_suffix is False:
                return f"Skipped {image_id} (ImageNet overlap)"
            
            url = f"https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
            filename = f"{image_id}__{label_suffix}.jpg"
            file_path = os.path.join(dest_dir, filename)
            
            # Skip if already exists
            if os.path.exists(file_path):
                return f"Skipped {image_id} (already exists)"
                
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                    
                return f"Downloaded {image_id}"
            except Exception as e:
                return f"Failed {image_id}: {str(e)}"
        
        # Download images in parallel
        total_images = len(image_to_labels)
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all download tasks
            future_to_id = {
                executor.submit(download_single_image_with_suffix, img_id, labels): img_id 
                for img_id, labels in image_to_labels.items()
            }
            
            # Process completed downloads
            successful = 0
            failed = 0
            imagenet_filtered = 0
            processed = 0
            
            for future in as_completed(future_to_id):
                result = future.result()
                processed += 1
                
                if "Downloaded" in result:
                    successful += 1
                elif "ImageNet overlap" in result:
                    imagenet_filtered += 1
                elif "Skipped" in result:
                    successful += 1
                else:
                    failed += 1
                
                # Print progress every 10 downloads or at the end
                if processed % 10 == 0 or processed == total_images:
                    print(f"Progress: {processed}/{total_images} processed, {successful} downloaded, {imagenet_filtered} filtered (ImageNet), {failed} failed")
        
        print(f"Final: {successful} downloaded, {imagenet_filtered} filtered out (ImageNet overlap), {failed} failed")
        return successful

    

    
    def create_image_list_file(self, image_ids: Set[str], split: str = 'train'):
        """Create image list file for downloader tool."""
        list_file = self.download_dir / f"{split}_subset_images.txt"
        
        with open(list_file, 'w') as f:
            for image_id in image_ids:
                f.write(f"{image_id}\n")
        
        print(f"Created image list file: {list_file}")
        return list_file
    
    def download_images_with_tool(self, image_ids: Set[str], split: str = 'train'):
        """
        Download images using the official OpenImages downloader tool.
        Requires: pip install openimages
        """
        try:
            import openimages.download as oi_download
            
            # Create list file
            list_file = self.create_image_list_file(image_ids, split)
            
            # Download using OpenImages tool
            print(f"Downloading {len(image_ids)} images...")
            oi_download.download_images(
                csv_dir=str(self.download_dir),
                image_list=str(list_file),
                image_dir=str(self.download_dir / split),
                num_processes=4  # Adjust based on your bandwidth/CPU
            )
            
        except ImportError:
            print("OpenImages downloader not found. Install with: pip install openimages")
    





if __name__ == "__main__":
    # Example usage for large-scale dataset creation using ONLY training split
    print("Creating natural diverse subset from training split...")
    
    # FOR 50K+ IMAGES: Use very relaxed parameters on training split only
    selected_images, downloader = create_natural_subset_for_ood(
        download_dir="./data/openimages",
        target_images=2000000,        # Target 50K images
        min_labels_per_class=1,     # Include ALL classes (minimum possible)
        max_labels_per_class=10000  # No practical limit on class size
    )
    
    print(f"\nSetup complete! Selected {len(selected_images)} images for download.")
    
    # Provide feedback on results
    if len(selected_images) < 50000:
        print(f"\nINFO: Got {len(selected_images)} images from training split (target: 50,000)")
        print("This is the maximum available from your current training data.")
        print(f"Your training set has 566K annotations → {len(selected_images)} unique images")
        print(f"Deduplication ratio: {len(selected_images)/566520*100:.1f}%")
        
        print("\nTo get more images:")
        print("1. Download the FULL OpenImages training set (9+ million annotations)")
        print("   This would provide 500K-1M+ unique images instead of your current subset")
        print("2. Your current data appears to be a subset/sample of the full OpenImages dataset")
        
    else:
        print(f"SUCCESS: Achieved target of {len(selected_images)} images!")
    
    print(f"\nEstimated download size: ~{len(selected_images) * 0.7:.1f} GB")
    
    # Show what parameters were most effective
    print(f"\n" + "="*60)
    print("TRAINING SPLIT ANALYSIS")
    print("="*60)
    print("Using most aggressive parameters:")
    print("- min_labels_per_class=1 (include ALL classes)")
    print("- max_labels_per_class=10000 (no class size limits)")
    print("- Only training split (as requested)")
    print(f"- Result: {len(selected_images)} unique images from 566K annotations")
    
    if len(selected_images) < 10000:
        print(f"\nWARNING: Very low image count suggests your training data is limited.")
        print("Consider downloading the full OpenImages training set for better results.")
