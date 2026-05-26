import logging
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple, Union, Dict, Any

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """Convolutional Block: Conv2D -> BatchNorm (optional) -> ReLU -> MaxPool2D."""
    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OncoNetCNN(nn.Module):
    """Custom CNN architecture designed from scratch for brain tumor classification."""
    def __init__(
        self, 
        num_classes: int = 4, 
        channels: List[int] = [32, 64, 128, 256],
        dropout_rate: float = 0.4,
        use_batch_norm: bool = True
    ):
        """
        Args:
            num_classes: Number of classification categories.
            channels: List of output channels for the 4 convolutional blocks.
            dropout_rate: Dropout probability for the classifier block.
            use_batch_norm: Whether to include Batch Normalization in the conv blocks.
        """
        super().__init__()
        assert len(channels) == 4, "OncoNetCNN requires exactly 4 channel configurations for its 4 blocks."
        
        # 4 Convolutional Blocks
        self.block1 = ConvBlock(3, channels[0], use_batch_norm=use_batch_norm)
        self.block2 = ConvBlock(channels[0], channels[1], use_batch_norm=use_batch_norm)
        self.block3 = ConvBlock(channels[1], channels[2], use_batch_norm=use_batch_norm)
        self.block4 = ConvBlock(channels[2], channels[3], use_batch_norm=use_batch_norm)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier: FC Layer -> Dropout -> Output Layer
        fc_hidden_dim = 128
        self.fc1 = nn.Linear(channels[3], fc_hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)
        
        # Initialize Weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Perform Kaiming (He) normal initialization on linear and conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_transfer_model(
    model_name: str, 
    num_classes: int = 4, 
    pretrained: bool = True, 
    fine_tune: bool = False
) -> nn.Module:
    """Loads and configures pre-trained models for transfer learning.

    Supported models:
    - resnet50
    - efficientnet_b0
    - densenet121

    Args:
        model_name: Name of model architecture.
        num_classes: Target number of output classes.
        pretrained: If True, load pre-trained ImageNet weights.
        fine_tune: If True, keep backbone weights trainable. If False, freeze backbone.
    """
    logger.info(f"Loading transfer learning model: {model_name} (pretrained={pretrained}, fine_tune={fine_tune})")
    
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        
        # Freeze or unfreeze parameters
        for param in model.parameters():
            param.requires_grad = fine_tune
            
        # Replace FC layer
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )
        # Ensure new classifier layers are trainable
        for param in model.fc.parameters():
            param.requires_grad = True
            
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        for param in model.parameters():
            param.requires_grad = fine_tune
            
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        
        for param in model.parameters():
            param.requires_grad = fine_tune
            
        # Replace classifier
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    else:
        raise ValueError(f"Unsupported transfer model: {model_name}. Choose from 'resnet50', 'efficientnet_b0', 'densenet121'.")
        
    return model


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary_string(model: nn.Module) -> str:
    """Generates a summary description of the model architecture."""
    summary_str = []
    summary_str.append("==================================================================")
    summary_str.append(f"Model Class: {model.__class__.__name__}")
    summary_str.append("==================================================================")
    summary_str.append(f"{'Layer Name':<40} {'Type':<20} {'Requires Grad':<15}")
    summary_str.append("------------------------------------------------------------------")
    for name, module in model.named_children():
        requires_grad = any(p.requires_grad for p in module.parameters())
        summary_str.append(f"{name:<40} {module.__class__.__name__:<20} {str(requires_grad):<15}")
    summary_str.append("------------------------------------------------------------------")
    summary_str.append(f"Total Parameters: {count_parameters(model):,}")
    summary_str.append(f"Trainable Parameters: {count_trainable_parameters(model):,}")
    summary_str.append("==================================================================")
    return "\n".join(summary_str)
