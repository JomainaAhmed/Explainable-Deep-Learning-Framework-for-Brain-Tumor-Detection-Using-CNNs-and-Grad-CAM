import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

logger = logging.getLogger(__name__)

# Set publication style plot parameters
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def plot_learning_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], 'o-', label="Train Loss", color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, history["val_loss"], 's--', label="Val Loss", color='#ff7f0e', linewidth=2)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(loc="upper right")
    
    # Accuracy Curve
    axes[1].plot(epochs, history["train_acc"], 'o-', label="Train Acc", color='#2ca02c', linewidth=2)
    axes[1].plot(epochs, history["val_acc"], 's--', label="Val Acc", color='#d62728', linewidth=2)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Learning curves saved to {save_path}")


def plot_roc_curve(y_true: List[int], y_probs: List[List[float]], class_names: List[str], save_path: str) -> None:
    """Plots One-vs-Rest ROC curve for each class."""
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)
    n_classes = len(class_names)
    
    # Binarize labels
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1.0
        
    plt.figure(figsize=(8, 6))
    
    # Plot curve for each class
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs_arr[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
    # Plot random reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Multi-Class')
    plt.legend(loc="lower right")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curves saved to {save_path}")


def plot_precision_recall_curve(y_true: List[int], y_probs: List[List[float]], class_names: List[str], save_path: str) -> None:
    """Plots Precision-Recall curve for each class."""
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)
    n_classes = len(class_names)
    
    # Binarize labels
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1.0
        
    plt.figure(figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_onehot[:, i], y_probs_arr[:, i])
        ap = average_precision_score(y_onehot[:, i], y_probs_arr[:, i])
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                 label=f'{class_names[i]} (AP = {ap:.3f})')
        
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"PR curves saved to {save_path}")


def plot_confusion_matrix(conf_matrix: List[List[int]], class_names: List[str], save_path: str) -> None:
    """Plots Confusion Matrix as a labeled heatmap."""
    matrix = np.array(conf_matrix)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Calculate cell percentages and annotations
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            percent = (val / matrix[i].sum()) * 100 if matrix[i].sum() > 0 else 0
            text = f"{val}\n({percent:.1f}%)"
            plt.text(j, i, text,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if val > thresh else "black")
            
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix plot saved to {save_path}")


def plot_learned_filters(model: nn.Module, save_path: str) -> None:
    """Visualizes learned convolution filters of the first layer of the network."""
    # Search for the first convolutional layer
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
            
    if first_conv is None:
        logger.warning("Could not find any Conv2D layer in the model for filter visualization.")
        return
        
    # Shape: (out_channels, in_channels, kh, kw)
    weights = first_conv.weight.detach().cpu().numpy()
    out_channels = weights.shape[0]
    
    # Normalize weights to [0, 1] for visualization
    w_min, w_max = weights.min(), weights.max()
    weights = (weights - w_min) / (w_max - w_min) if w_max > w_min else weights
    
    # We display up to 16 channels in a 4x4 grid
    num_filters = min(16, out_channels)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(num_filters):
        filter_img = weights[i]
        # Transpose to (kh, kw, in_channels) for plotting
        filter_img = np.transpose(filter_img, (1, 2, 0))
        
        # If input has 3 channels, plot as RGB. If 1, plot as grayscale
        if filter_img.shape[2] == 3:
            axes[i].imshow(filter_img)
        else:
            axes[i].imshow(filter_img[:, :, 0], cmap='gray')
            
        axes[i].axis('off')
        axes[i].set_title(f"Filter {i+1}", fontsize=9)
        
    for j in range(num_filters, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle("Learned First-Layer Convolution Filters")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"First-layer filter plot saved to {save_path}")


def plot_feature_maps(model: nn.Module, input_tensor: torch.Tensor, save_path: str) -> None:
    """Feeds a sample through the custom model and visualizes outputs from early blocks."""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Extract features using CerebroNetCNN blocks (checks model class name)
    try:
        with torch.no_grad():
            if hasattr(model, "block1"):
                # Custom CNN
                fmap1 = model.block1(input_tensor)
                fmap2 = model.block2(fmap1)
            elif hasattr(model, "features"):
                # ResNet / DenseNet / EfficientNet
                # Just extract from the very first convolutional features layer
                if isinstance(model.features, nn.Sequential):
                    fmap1 = model.features[0](input_tensor)
                    fmap2 = model.features[3](fmap1)
                else:
                    # Generic torchvision models
                    fmap1 = model.features(input_tensor)
                    fmap2 = fmap1
            else:
                logger.warning("Unrecognized model structure for feature maps.")
                return
                
        # Move map to cpu
        features = fmap1.detach().cpu().numpy()[0] # Shape: (C, H, W)
        num_maps = min(8, features.shape[0])
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for idx in range(num_maps):
            f_slice = features[idx]
            # Normalize map for display
            fs_min, fs_max = f_slice.min(), f_slice.max()
            f_slice = (f_slice - fs_min) / (fs_max - fs_min) if fs_max > fs_min else f_slice
            
            axes[idx].imshow(f_slice, cmap='viridis')
            axes[idx].axis('off')
            axes[idx].set_title(f"Channel {idx+1}")
            
        plt.suptitle("Intermediate Layer Feature Map Activations")
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature maps plot saved to {save_path}")
        
    except Exception as e:
        logger.warning(f"Failed to plot intermediate feature maps: {e}")


def plot_gradcam_grid(
    original_image: np.ndarray,
    predicted_class: str,
    confidence: float,
    heatmap: np.ndarray,
    blended: np.ndarray,
    save_path: str
) -> None:
    """Generates an explainability grid plot showing MRI inputs, Grad-CAM maps, and overlays."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original MRI Scan")
    axes[0].axis("off")
    
    # Heatmap
    im1 = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Blended
    axes[2].imshow(blended)
    axes[2].set_title(f"Overlay\nPredicted: {predicted_class} ({confidence*100:.1f}%)")
    axes[2].axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Grad-CAM grid plot saved to {save_path}")


def plot_ablation_comparison(ablation_results: Dict[str, Any], save_path: str) -> None:
    """Plots comparisons of validation Accuracy and F1 Scores across ablation model variants."""
    models = list(ablation_results.keys())
    names = [res["name"] for res in ablation_results.values()]
    accuracies = [res["val_accuracy"] * 100 for res in ablation_results.values()]
    f1_scores = [res["val_f1"] * 100 for res in ablation_results.values()]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#1f77b4')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#2ca02c')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Ablation Study: Architecture Variations')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    
    # Annotations
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Ablation study comparison plot saved to {save_path}")


def plot_hyperparameter_comparison(results: List[Dict[str, Any]], save_path: str) -> None:
    """Plots top hyperparameter configurations vs validation accuracy."""
    sorted_res = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)[:6]
    
    labels = []
    accuracies = []
    for r in sorted_res:
        label = f"{r['optimizer']}\nLR={r['lr']}\nBS={r['batch_size']}"
        labels.append(label)
        accuracies.append(r["val_accuracy"] * 100)
        
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, accuracies, color='#673ab7', width=0.5)
    
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Top 6 Hyperparameter Configurations')
    plt.ylim(0, 110)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Hyperparameter comparison plot saved to {save_path}")
