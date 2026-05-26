import os
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple

from src.metrics import calculate_metrics
from src.dataset import CLASSES

logger = logging.getLogger(__name__)

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str = "outputs"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Runs model evaluation on test dataset.

    Returns:
        A tuple of:
            - metrics: Dict of evaluation scores (accuracy, f1, precision, recall, confusion_matrix, etc.)
            - misclassified: List of dicts detailing misclassified images for error analysis.
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    misclassified = []
    
    # Track paths/samples if needed for plotting
    sample_idx = 0
    
    with torch.no_grad():
        for images, labels, original_imgs in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Move to cpu
            preds_cpu = predicted.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            probs_cpu = probs.cpu().numpy()
            
            all_preds.extend(preds_cpu.tolist())
            all_labels.extend(labels_cpu.tolist())
            all_probs.extend(probs_cpu.tolist())
            
            # Error Analysis tracking
            for i in range(images.size(0)):
                pred = preds_cpu[i]
                true_label = labels_cpu[i]
                
                if pred != true_label:
                    misclassified.append({
                        "index": sample_idx,
                        "true_class": CLASSES[true_label],
                        "pred_class": CLASSES[pred],
                        "confidence": float(probs_cpu[i][pred]),
                        "original_image": original_imgs[i].numpy() if isinstance(original_imgs[i], torch.Tensor) else original_imgs[i]
                    })
                sample_idx += 1
                
    metrics = calculate_metrics(all_labels, all_preds, all_probs, CLASSES)
    
    # Save error analysis plots
    if len(misclassified) > 0 and output_dir:
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_misclassified(misclassified, os.path.join(plot_dir, "error_analysis.png"))
        
    return metrics, misclassified


def plot_misclassified(misclassified: List[Dict[str, Any]], save_path: str, max_plots: int = 8) -> None:
    """Generates and saves a grid collage of misclassified examples.

    Args:
        misclassified: List of error analysis dictionaries containing original images, labels, predictions.
        save_path: Location to save the output plot image.
        max_plots: Maximum number of plots to display in the grid (default: 8).
    """
    num_to_show = min(len(misclassified), max_plots)
    if num_to_show == 0:
        return
        
    cols = min(4, num_to_show)
    rows = (num_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for idx in range(num_to_show):
        item = misclassified[idx]
        ax = axes[idx]
        
        # Display image
        ax.imshow(item["original_image"])
        ax.axis("off")
        
        # Set titles with class names
        title = f"True: {item['true_class']}\nPred: {item['pred_class']} ({item['confidence'] * 100:.1f}%)"
        ax.set_title(title, fontsize=10, color='red')
        
    # Clean up empty subplots
    for idx in range(num_to_show, len(axes)):
        axes[idx].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Misclassified examples plot saved to {save_path}")
