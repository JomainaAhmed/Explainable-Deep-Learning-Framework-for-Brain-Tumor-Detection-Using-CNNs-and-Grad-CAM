import logging
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)

def calculate_metrics(
    y_true: List[int], 
    y_pred: List[int], 
    y_probs: List[List[float]], 
    class_names: List[str]
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        y_probs: Predicted class probabilities of shape (N, C).
        class_names: List of class names corresponding to indices.

    Returns:
        A dictionary containing:
            - accuracy: Overall accuracy.
            - precision: Macro-averaged precision.
            - recall: Macro-averaged recall.
            - f1: Macro-averaged F1 score.
            - roc_auc: Macro-averaged One-vs-Rest ROC-AUC.
            - confusion_matrix: 2D array representation.
            - per_class: Dictionary containing precision, recall, F1 for each class.
            - raw_report: Classification report string.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_probs_arr = np.array(y_probs)

    # Basic metrics
    acc = accuracy_score(y_true_arr, y_pred_arr)
    
    # Precision, Recall, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average='macro', zero_division=0
    )
    
    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average=None, zero_division=0
    )
    
    per_class_metrics = {}
    for idx, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1": float(f1_class[idx]),
            "support": int(support_class[idx])
        }

    # ROC-AUC (requires probabilities)
    try:
        if y_probs_arr.shape[1] > 1:
            # Multi-class ROC-AUC One-vs-Rest
            auc = roc_auc_score(y_true_arr, y_probs_arr, multi_class='ovr', average='macro')
        else:
            auc = 0.5
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}. Defaulting to 0.5.")
        auc = 0.5

    conf_mat = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))
    
    raw_report = classification_report(
        y_true_arr, y_pred_arr, target_names=class_names, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1": float(f1_macro),
        "roc_auc": float(auc),
        "confusion_matrix": conf_mat.tolist(),
        "per_class": per_class_metrics,
        "raw_report": raw_report
    }
