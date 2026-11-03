import os
import cv2
import shutil
import random
import logging
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from torch.utils.data import Dataset
from src.utils import dataframe_to_markdown

logger = logging.getLogger(__name__)

# Fixed classes for the framework
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]

class BrainMRIDataset(Dataset):
    """PyTorch Dataset for Brain Tumor MRI Classification."""
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Path to the split directory (e.g. data/processed/train)
            transform: Albumentations transforms to apply.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(CLASSES)}

        # Scan for images
        for cls in CLASSES:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            logger.warning(f"No MRI images found in {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, np.ndarray]:
        """Returns:
            - transformed_image: float tensor (or array) ready for model forward pass.
            - label: integer index of the tumor class.
            - original_image: uint8 RGB numpy array, useful for visualization/Grad-CAM.
        """
        img_path, label = self.samples[idx]
        # Read as BGR, convert to RGB
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            raise IOError(f"Could not load image: {img_path}")
        
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=rgb_img)
            img_tensor = augmented["image"]
        else:
            # Fallback simple normalization
            img_tensor = rgb_img.astype(np.float32) / 255.0

        return img_tensor, label, rgb_img


def download_real_dataset(dest_dir: str) -> bool:
    """Attempts to download a public brain tumor dataset.
    Returns:
        bool: True if successful, False otherwise.
    """
    os.makedirs(dest_dir, exist_ok=True)
    # We can try to download from a stable direct URL, e.g. a public release on GitHub or figshare
    # For robust verification and speed, we support this but default to synthetic if it fails or if offline
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"
    zip_path = os.path.join(dest_dir, "brain-tumor.zip")
    
    logger.info("Attempting to download public brain tumor dataset...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        logger.info("Download completed. Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        os.remove(zip_path)
        logger.info("Extraction complete.")
        return True
    except Exception as e:
        logger.warning(f"Failed to download public dataset from internet: {e}")
        return False


def generate_synthetic_mri_slice(cls: str, img_size: int = 224) -> np.ndarray:
    """Generates a highly-structured synthetic 2D MRI brain slice.
    
    Contains:
    - Dark background
    - Gray skull / brain contour
    - Symmetrical ventricles (butterfly shapes)
    - Hemispheric dividing line
    - Pathologically-accurate tumor structures for tumor classes (edema, location).
    """
    # 1. Blank grayscale image
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    center = img_size // 2
    
    # 2. Draw skull outline
    skull_radius_x = int(img_size * 0.40)
    skull_radius_y = int(img_size * 0.43)
    cv2.ellipse(img, (center, center), (skull_radius_x, skull_radius_y), 0, 0, 360, 40, -1)
    
    # 3. Draw brain tissue (slightly smaller than skull)
    brain_radius_x = int(img_size * 0.37)
    brain_radius_y = int(img_size * 0.40)
    cv2.ellipse(img, (center, center), (brain_radius_x, brain_radius_y), 0, 0, 360, 75, -1)
    
    # 4. Central division line (longitudinal fissure)
    cv2.line(img, (center, center - brain_radius_y), (center, center + brain_radius_y), 45, 1)
    
    # 5. Ventricles (darker regions near center, butterfly shape)
    v_width = int(img_size * 0.08)
    v_height = int(img_size * 0.12)
    # Left ventricle
    cv2.ellipse(img, (center - int(v_width*0.8), center), (v_width, v_height), 15, 0, 360, 25, -1)
    # Right ventricle
    cv2.ellipse(img, (center + int(v_width*0.8), center), (v_width, v_height), -15, 0, 360, 25, -1)
    
    # Apply some blurring to ventricles for transition
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 6. Add Tumor mass depending on class
    if cls == "glioma":
        # Gliomas are infiltrative, located within the cerebral hemispheres, with blurred edges (edema)
        # Position: Left upper cerebrum
        tx, ty = center - int(img_size * 0.18), center - int(img_size * 0.15)
        r = int(img_size * 0.09)
        # Edema (surrounding halo, low brightness spike)
        cv2.circle(img, (tx, ty), r + int(img_size * 0.04), 105, -1)
        # Core tumor (irregular/rough)
        cv2.circle(img, (tx, ty), r, 190, -1)
        # Some texture
        cv2.circle(img, (tx + int(r*0.3), ty - int(r*0.2)), int(r*0.4), 225, -1)
        cv2.circle(img, (tx - int(r*0.2), ty + int(r*0.3)), int(r*0.3), 130, -1)
        
    elif cls == "meningioma":
        # Meningiomas are extra-axial, well-defined, bright, and sit near the skull boundary
        # Position: Right outer boundary
        tx, ty = center + int(brain_radius_x * 0.8), center - int(brain_radius_y * 0.4)
        r = int(img_size * 0.07)
        # Sharp edge tumor
        cv2.circle(img, (tx, ty), r, 220, -1)
        # Bright hyperintense center
        cv2.circle(img, (tx, ty), int(r * 0.6), 245, -1)
        # Dural tail simulation (thin line along the edge)
        cv2.ellipse(img, (center, center), (brain_radius_x, brain_radius_y), 0, -60, -30, 180, 2)
        
    elif cls == "pituitary":
        # Pituitary tumors sit in the cellar region, bottom center of the brain
        # Position: Base/Bottom center
        tx, ty = center, center + int(brain_radius_y * 0.65)
        r = int(img_size * 0.065)
        # Well-demarcated round mass
        cv2.circle(img, (tx, ty), r, 200, -1)
        cv2.circle(img, (tx, ty), int(r * 0.5), 235, -1)
        # Optic chiasm displacement simulation (dark compression lines nearby)
        cv2.line(img, (tx - r - 5, ty - 5), (tx + r + 5, ty - 5), 35, 1)

    # 7. Apply general gaussian blur to merge tumor borders
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 8. Add realistic MRI noise
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    mri_slice = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 9. Convert to 3-channel (RGB) as expected by ImageNet transfer learning models
    mri_rgb = cv2.merge([mri_slice, mri_slice, mri_slice])
    return mri_rgb


def generate_synthetic_mri_dataset(dest_dir: str, num_samples_per_class: int = 100, img_size: int = 224) -> None:
    """Generates a complete synthetic dataset structured into class folders."""
    logger.info(f"Generating synthetic MRI dataset in {dest_dir} ({num_samples_per_class} per class)...")
    
    # Reset/Create directory
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    
    for cls in CLASSES:
        cls_dir = os.path.join(dest_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(num_samples_per_class):
            img = generate_synthetic_mri_slice(cls, img_size=img_size)
            filename = f"{cls}_{i:04d}.png"
            cv2.imwrite(os.path.join(cls_dir, filename), img)
            
    logger.info("Synthetic MRI dataset generation complete.")


def split_dataset(raw_dir: str, processed_dir: str, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> None:
    """Splits raw class folders into train/val/test splits."""
    logger.info(f"Splitting raw dataset from {raw_dir} into {processed_dir} with splits {split_ratios}...")
    
    # Remove existing split if it exists
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
        
    train_ratio, val_ratio, test_ratio = split_ratios
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)
            
    for cls in CLASSES:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            logger.warning(f"Class folder {cls_dir} does not exist.")
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Copy helper
        def copy_files(file_list, split_name):
            for filename in file_list:
                src_path = os.path.join(cls_dir, filename)
                dst_path = os.path.join(processed_dir, split_name, cls, filename)
                shutil.copy2(src_path, dst_path)
                
        copy_files(train_imgs, "train")
        copy_files(val_imgs, "val")
        copy_files(test_imgs, "test")
        
    logger.info("Dataset split complete.")


def generate_dataset_statistics(processed_dir: str, output_report_path: str) -> Dict[str, Any]:
    """Generates statistics and distributions of the dataset splits and saves a summary report."""
    logger.info("Generating dataset statistics and distributions...")
    stats = {}
    
    records = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(processed_dir, split)
        split_stats = {}
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            count = len(os.listdir(cls_dir)) if os.path.exists(cls_dir) else 0
            split_stats[cls] = count
            records.append({"Split": split, "Class": cls, "Count": count})
        stats[split] = split_stats
        
    df = pd.DataFrame(records)
    
    # Calculate pixel mean and standard deviation across train split
    train_dir = os.path.join(processed_dir, "train")
    pixels = []
    
    # Sample up to 100 images to calculate approximate dataset statistics (prevents memory issues)
    sample_count = 0
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, f)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    pixels.append(img.flatten())
                    sample_count += 1
                if sample_count >= 100:
                    break
        if sample_count >= 100:
            break
            
    if pixels:
        all_pixels = np.concatenate(pixels)
        mean_val = all_pixels.mean() / 255.0
        std_val = all_pixels.std() / 255.0
    else:
        mean_val, std_val = 0.5, 0.5
        
    stats["pixel_mean"] = float(mean_val)
    stats["pixel_std"] = float(std_val)
    
    # Create Markdown report
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("# CEREBRONET Dataset Statistics Report\n\n")
        f.write("This report summarizes the distribution and statistics of the brain tumor MRI scans.\n\n")
        
        f.write("## Class Distribution Table\n\n")
        pivot_df = df.pivot(index="Class", columns="Split", values="Count")
        pivot_df["Total"] = pivot_df.sum(axis=1)
        f.write(dataframe_to_markdown(pivot_df))
        f.write("\n\n")
        
        f.write("## Data Summary Details\n\n")
        f.write(f"- **Total Training Images**: {pivot_df['train'].sum()}\n")
        f.write(f"- **Total Validation Images**: {pivot_df['val'].sum()}\n")
        f.write(f"- **Total Test Images**: {pivot_df['test'].sum()}\n")
        f.write(f"- **Total Images**: {pivot_df['Total'].sum()}\n")
        f.write(f"- **Calculated Image Mean (0-1 range)**: {mean_val:.4f}\n")
        f.write(f"- **Calculated Image Std (0-1 range)**: {std_val:.4f}\n\n")
        
        f.write("### Splits Allocation percentages:\n")
        total_count = pivot_df["Total"].sum()
        f.write(f"- Train: {pivot_df['train'].sum() / total_count * 100:.1f}%\n")
        f.write(f"- Validation: {pivot_df['val'].sum() / total_count * 100:.1f}%\n")
        f.write(f"- Test: {pivot_df['test'].sum() / total_count * 100:.1f}%\n")
        
    logger.info(f"Dataset stats report saved to {output_report_path}")
    return stats
