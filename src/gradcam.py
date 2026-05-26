import cv2
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

class GradCAM:
    """Custom implementation of Gradient-weighted Class Activation Mapping (Grad-CAM)."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The PyTorch neural network model.
            target_layer: The target convolutional layer (usually the last conv layer).
        """
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.handlers = []

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture activations and gradients of the target layer."""
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple, index 0 corresponds to gradient w.r.t layer output
            self.gradients = grad_output[0]

        # Register forward hook
        h_f = self.target_layer.register_forward_hook(forward_hook)
        self.handlers.append(h_f)

        # Register backward hook (using modern full_backward_hook to avoid deprecation issues)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            h_b = self.target_layer.register_full_backward_hook(backward_hook)
        else:
            # Fallback for older torch versions
            h_b = self.target_layer.register_backward_hook(backward_hook)
        self.handlers.append(h_b)

    def remove_hooks(self):
        """Remove hooks to prevent memory leaks and issues with subsequent runs."""
        for handler in self.handlers:
            handler.remove()
        self.handlers.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_heatmap(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generates the raw normalized 2D Grad-CAM heatmap.

        Args:
            input_tensor: PyTorch tensor input image of shape (1, C, H, W).
            target_class: Target class index. If None, uses the class with highest probability.

        Returns:
            Normalized 2D numpy array heatmap of shape (H_layer, W_layer) with values in range [0, 1].
        """
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        # Target score
        score = logits[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        if self.feature_maps is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hook execution failed. Check if target_layer is correctly placed in the forward pass."
            )
            
        # Extract features and gradients
        # Shape of feature_maps: (1, C_conv, H_conv, W_conv)
        features = self.feature_maps.detach().cpu().numpy()[0]
        # Shape of gradients: (1, C_conv, H_conv, W_conv)
        grads = self.gradients.detach().cpu().numpy()[0]
        
        # Compute weights: Global Average Pool of gradients
        # Shape: (C_conv,)
        weights = np.mean(grads, axis=(1, 2))
        
        # Compute weighted activation map
        # Shape: (H_conv, W_conv)
        heatmap = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * features[i]
            
        # Apply ReLU (only positive contributions)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap to [0, 1] range
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        return heatmap


def overlay_heatmap(
    heatmap: np.ndarray, 
    original_image_rgb: np.ndarray, 
    alpha: float = 0.4, 
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlays the heatmap on the original MRI scan.

    Args:
        heatmap: 2D numpy array of shape (H_conv, W_conv) in [0, 1].
        original_image_rgb: 3D uint8 numpy array representing the original image (H, W, 3) in RGB.
        alpha: Blending weight for the heatmap overlay (0.0 = only original image, 1.0 = only heatmap).
        colormap: OpenCV colormap index (e.g. cv2.COLORMAP_JET).

    Returns:
        A tuple of:
            - color_heatmap: Colorized heatmap (RGB).
            - blended_image: Heatmap overlayed on the original image (RGB).
    """
    h, w, _ = original_image_rgb.shape
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Scale heatmap to [0, 255] and convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply OpenCV colormap (outputs BGR)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert colormapped heatmap to RGB
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)
    
    # Blend images
    blended = cv2.addWeighted(heatmap_colored_rgb, alpha, original_image_rgb, 1.0 - alpha, 0)
    
    return heatmap_colored_rgb, blended


def get_last_conv_layer(model_name: str, model: nn.Module) -> nn.Module:
    """Dynamically resolves and returns the last convolutional layer reference.

    Args:
        model_name: Name of model ('custom', 'resnet50', 'efficientnet_b0', 'densenet121').
        model: PyTorch model object.
    """
    if model_name == "custom":
        # OncoNetCNN: target layer is the convolutional layer inside Block 4
        # self.block4 is a ConvBlock which contains a Sequential: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        return model.block4.block[0]
        
    elif model_name == "resnet50":
        # ResNet50: target is the third conv layer of the last bottleneck in layer 4
        return model.layer4[-1].conv3
        
    elif model_name == "efficientnet_b0":
        # EfficientNet-B0: target is the convolutional layer in the final features block (features[8])
        # features[8] is a Conv2dNormActivation: Conv2d -> BatchNorm2d -> SiLU
        return model.features[8][0]
        
    elif model_name == "densenet121":
        # DenseNet121: target is the final conv layer of the final dense block
        # features.denseblock4 is a Sequential, denselayer16 is the last layer, which has conv2
        return model.features.denseblock4.denselayer16.conv2
        
    else:
        # Fallback to search recursively for the last Conv2D layer in the architecture
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            logger.info("Found last convolutional layer automatically via module scan.")
            return last_conv
        raise ValueError(f"Could not identify target conv layer for model '{model_name}'.")
