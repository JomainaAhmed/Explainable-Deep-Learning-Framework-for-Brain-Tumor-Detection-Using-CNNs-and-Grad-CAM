import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_train_transforms(
    img_size: int = 224, 
    use_aug: bool = True,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> A.Compose:
    """Returns the training data transformation pipeline.

    Args:
        img_size: Dimension to resize images to (img_size x img_size).
        use_aug: If True, include strong data augmentations (flips, rotates, brightness/contrast, noise).
                 If False, only apply resize, normalize, and Tensor conversion.
        mean: Normalization channel means.
        std: Normalization channel standard deviations.
    """
    if use_aug:
        transform_list = [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    else:
        transform_list = [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
        
    return A.Compose(transform_list)


def get_val_test_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> A.Compose:
    """Returns the validation and test data transformation pipeline.

    Only applies resizing, normalization, and conversion to PyTorch Tensor.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_experiment_transforms(
    aug_type: str,
    img_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> A.Compose:
    """Returns a training transform pipeline containing only ONE specific type of augmentation.
    
    Used to evaluate the isolated impact of individual augmentations (e.g. rotation, flip).
    """
    base_ops = [A.Resize(height=img_size, width=img_size)]
    
    if aug_type == "rotation":
        base_ops.append(A.Rotate(limit=45, p=1.0))
    elif aug_type == "horizontal_flip":
        base_ops.append(A.HorizontalFlip(p=1.0))
    elif aug_type == "vertical_flip":
        base_ops.append(A.VerticalFlip(p=1.0))
    elif aug_type == "contrast":
        base_ops.append(A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.3, p=1.0))
    elif aug_type == "brightness":
        base_ops.append(A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.0, p=1.0))
    elif aug_type == "noise":
        base_ops.append(A.GaussNoise(var_limit=(40.0, 50.0), p=1.0))
        
    base_ops.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return A.Compose(base_ops)
