import os
import argparse
import logging
import cv2
import torch
import numpy as np
from typing import Dict, Any, Tuple

from src.config import load_config
from src.model import CerebroNetCNN, OncoNetCNN, get_transfer_model, get_model_summary_string
from src.transforms import get_val_test_transforms
from src.gradcam import GradCAM, overlay_heatmap, get_last_conv_layer
from src.dataset import CLASSES

logger = logging.getLogger(__name__)

def load_inference_model(
    model_name: str, 
    checkpoint_path: str, 
    device: torch.device
) -> torch.nn.Module:
    """Loads a model architecture and restores weights from a checkpoint path."""
    logger.info(f"Loading model '{model_name}' for inference from {checkpoint_path}...")
    
    if model_name == "custom":
        model = CerebroNetCNN(num_classes=len(CLASSES))
    else:
        model = get_transfer_model(model_name, num_classes=len(CLASSES), pretrained=False)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("Model loaded and set to eval mode.")
    return model


def run_single_inference(
    model: torch.nn.Module,
    model_name: str,
    image_path: str,
    device: torch.device,
    img_size: int = 224
) -> Tuple[str, float, np.ndarray]:
    """Runs a forward pass on a single image and returns predictions.

    Returns:
        A tuple of:
            - pred_class: Name of predicted class.
            - confidence: Probability score of the predicted class.
            - original_image_rgb: The read image array in RGB.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise IOError(f"Failed to read image at {image_path}")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    # Preprocessing
    transforms = get_val_test_transforms(img_size=img_size)
    augmented = transforms(image=rgb_img)
    input_tensor = augmented["image"].unsqueeze(0).to(device) # Add batch dimension (1, C, H, W)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    pred_class = CLASSES[pred_idx]
    
    return pred_class, confidence, rgb_img


def run_inference_with_gradcam(
    model: torch.nn.Module,
    model_name: str,
    image_path: str,
    device: torch.device,
    save_overlay_path: Optional[str] = None,
    img_size: int = 224
) -> Dict[str, Any]:
    """Runs inference and generates a Grad-CAM explainability heatmap overlay.

    Args:
        model: Loaded PyTorch model.
        model_name: Model identifier (e.g. 'custom', 'resnet50').
        image_path: Location of the input MRI scan.
        device: CPU or GPU device target.
        save_overlay_path: Target file path to write the overlay output image.
        img_size: Dimensions to resize input image to.

    Returns:
        A dictionary containing prediction class, confidence, raw heatmap, and blended image.
    """
    # 1. Run standard inference to get prediction
    pred_class, confidence, rgb_img = run_single_inference(model, model_name, image_path, device, img_size)
    pred_idx = CLASSES.index(pred_class)
    
    # 2. Setup input tensor again for gradient tracking
    transforms = get_val_test_transforms(img_size=img_size)
    augmented = transforms(image=rgb_img)
    input_tensor = augmented["image"].unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    
    # 3. Apply Grad-CAM
    target_layer = get_last_conv_layer(model_name, model)
    with GradCAM(model, target_layer) as gcam:
        # Note: generate_heatmap triggers backward() internally, so do not run under torch.no_grad()
        heatmap = gcam.generate_heatmap(input_tensor, target_class=pred_idx)
        
    # 4. Generate overlay
    colored_heatmap, blended = overlay_heatmap(heatmap, rgb_img, alpha=0.4)
    
    # 5. Optionally save result
    if save_overlay_path:
        os.makedirs(os.path.dirname(save_overlay_path), exist_ok=True)
        # Convert RGB to BGR for opencv write
        bgr_blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        
        # Add labels on the saved image
        h, w, _ = bgr_blended.shape
        label_text = f"Pred: {pred_class} ({confidence * 100:.1f}%)"
        cv2.putText(bgr_blended, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imwrite(save_overlay_path, bgr_blended)
        logger.info(f"Grad-CAM overlay saved to {save_overlay_path}")
        
    return {
        "class": pred_class,
        "confidence": confidence,
        "heatmap": heatmap,
        "blended": blended,
        "original_image": rgb_img
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEREBRONET Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input MRI image file")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "resnet50", "efficientnet_b0", "densenet121"], help="Model type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="outputs/gradcam/inference_overlay.png", help="Path to save output overlay plot")
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "auto"], help="Device to use")
    
    args = parser.parse_args()
    
    # Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
    
    # Device
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
        
    try:
        loaded_model = load_inference_model(args.model, args.checkpoint, dev)
        res = run_inference_with_gradcam(loaded_model, args.model, args.image, dev, save_overlay_path=args.output)
        
        print("\n===========================================")
        print("INFERENCE RESULTS:")
        print("===========================================")
        print(f"Input Image: {args.image}")
        print(f"Predicted Class: {res['class'].upper()}")
        print(f"Confidence Score: {res['confidence'] * 100:.2f}%")
        print(f"Grad-CAM output written to: {args.output}")
        print("===========================================\n")
        
    except Exception as e:
        logger.error(f"Inference run failed: {e}", exc_info=True)
