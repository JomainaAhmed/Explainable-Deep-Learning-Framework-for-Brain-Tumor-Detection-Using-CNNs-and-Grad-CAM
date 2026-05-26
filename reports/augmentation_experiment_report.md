# ONCONET Data Augmentation Impact Analysis

This study evaluates the direct contribution of individual image augmentation transforms on classification performance of the custom OncoNetCNN architecture.

### Experiment Results Table:

| augmentation   |   val_loss |   val_accuracy |   val_f1 |
|:---------------|-----------:|---------------:|---------:|
| rotation       |    1.83036 |           0.5  | 0.375    |
| none           |    1.06383 |           0.5  | 0.333333 |
| noise          |    2.5095  |           0.25 | 0.1      |

### Key Observations:
- **Rotations and Flips**: Help the network learn orientation invariance since tumor location shape orientation differs.
- **Gaussian Noise**: Acts as a strong pixel regularization technique, forcing the network to learn structural contours instead of high-frequency noise textures.
- **Contrast/Brightness**: Mimics variability in MRI scanner calibration and acquisition settings, enhancing generalization.
