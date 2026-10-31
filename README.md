# CEREBRONET: Explainable Deep Learning Framework for Brain Tumor Detection

CerebroNet is a complete, modular, and publication-ready deep learning framework designed to detect brain tumors from MRI scans with clinical explainability. It implements a custom Convolutional Neural Network (CNN) from scratch in PyTorch, compares performance with modern transfer learning backbones (ResNet, EfficientNet, DenseNet), and integrates a from-scratch Grad-CAM explainability module to visualize model focus areas.

---

## 🌟 Key Features
- **Custom Neural Architecture**: A modular 4-block CNN (`CerebroNetCNN`) built in PyTorch, featuring configurable channels, batch normalization, and dropout.
- **Deep Explainability**: Custom-built **Grad-CAM** implementation (no third-party Grad-CAM libraries) that intercepts forward feature maps and backward gradient pathways to map model attention to anatomical coordinates.
- **Transfer Learning Suite**: Integration and comparison wrappers for pre-trained `ResNet50`, `EfficientNet-B0`, and `DenseNet121` backbones.
- **Automated Experiment Runners**:
  - **Hyperparameter Grid Search**: Sweeping learning rates, batch sizes, and optimization algorithms.
  - **Ablation Studies**: Evaluation of validation performance when Batch Normalization, Dropout, or Data Augmentation are removed.
  - **Augmentation Impact Analysis**: Isolated evaluation of rotation, flips, contrast, and noise regularizers.
- **Unified Medical Dataset Support**: Supports automated download of public Brain MRI datasets alongside an advanced **Synthetic MRI Generator** fallback to ensure local reproducibility on any system (CPU/GPU).
- **Publication-Quality Reports**: Compiles ROC-AUC curves, Precision-Recall curves, confusion matrices, filters, feature maps, and detailed Markdown research logs.

---

## 📂 Repository Structure

```
OncoNet/
├── configs/
│   ├── base_config.yaml           # Primary training/model settings
│   └── tuning_config.yaml         # Config for HP tuning, augmentations, and ablations
├── data/
│   ├── raw/                       # Raw downloads / synthetic images
│   └── processed/                 # Train / Val / Test (70/15/15) split images
├── checkpoints/                   # Trained model weights (.pth)
├── outputs/
│   ├── plots/                     # Precision-Recall, ROC, learning curves, error grids
│   ├── gradcam/                   # Grad-CAM transparent heatmap overlays
│   └── features/                  # Conv filters and activation maps
├── reports/
│   ├── dataset_statistics.md      # Auto-generated dataset distribution reports
│   └── research_report.md         # Full research summary including experiment results
├── src/
│   ├── config.py                  # YAML Configuration parser
│   ├── dataset.py                 # PyTorch Dataset & Synthetic generator
│   ├── transforms.py              # Albumentations transforms setup
│   ├── model.py                   # CerebroNetCNN and transfer model architectures
│   ├── train.py                   # Custom mixed-precision PyTorch trainer
│   ├── evaluate.py                # Test set evaluator & error analysis
│   ├── inference.py               # Single scan inference & Grad-CAM visualizer CLI
│   ├── gradcam.py                 # Core hook-based Grad-CAM class
│   ├── visualization.py           # Publication-quality plotting utilities
│   ├── metrics.py                 # Performance metrics calculation (Scikit-learn)
│   └── utils.py                   # Reproducibility seed and logger settings
├── requirements.txt               # Dependencies list
├── main.py                        # Master script for train, evaluate, and Grad-CAM
└── run_experiments.py             # Script for HPO, Ablations, and comparisons
```

---

## 🛠️ Installation & Setup

1. **Clone and Navigate**:
   ```bash
   git clone https://github.com/yourusername/OncoNet.git
   cd OncoNet
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate       # Windows PowerShell/CMD
   source .venv/bin/activate    # Linux/MacOS
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Guide

### 1. Data Preparation
Prepare the dataset splits and write the statistics report. If no local dataset is found, it automatically generates a high-quality synthetic MRI brain dataset (representing Glioma, Meningioma, Pituitary, and normal scans):
```bash
python main.py --prepare --synthetic --num_samples 200
```
*Note: To download the public dataset from the web directly, run without the `--synthetic` flag.*

### 2. Model Training
Train the custom `CerebroNetCNN` model with default parameters (mixed-precision enabled on CUDA, early stopping enabled):
```bash
python main.py --train --model custom --epochs 15
```

### 3. Evaluation & Visual Metrics
Test the model on the test dataset split, outputting precision, recall, F1, and saving Confusion Matrices, ROC curves, PR curves, and a collage of misclassified scans (Error Analysis):
```bash
python main.py --evaluate --model custom
```

### 4. Explainability (Grad-CAM)
Produce and save Grad-CAM grids (Original Scan, Raw Heatmap, Blended Overlay) for each tumor class from the test set:
```bash
python main.py --gradcam --model custom
```
All outputs are saved to `outputs/gradcam/`.

### 5. Intermediate Features Visualization
Plot learned convolution filters of the first layer and activation feature maps from early layers:
```bash
python main.py --features --model custom
```
Plots are saved to `outputs/features/`.

### 6. Run Complete Experiments Suite
Execute the entire hyperparameter tuning sweep, ablation study, data augmentation tests, and transfer learning comparisons with a single command:
```bash
python run_experiments.py --epochs 3 --quick_run
```
*Remove `--quick_run` to execute a comprehensive parameter grid search. All tables and plots are automatically aggregated and formatted into the final research report: `reports/research_report.md`.*

---

## 🧠 Explainability: How Custom Grad-CAM Works
ONCONET does not use third-party libraries for Grad-CAM. It registers PyTorch hooks directly:
1. **Forward Hook**: Captures the final feature map activation tensor $A$ of the target convolutional layer (e.g. Block 4's Conv2d layer).
2. **Backward Hook**: Intercepts the backpropagated gradient tensor $G = \frac{\partial Y^c}{\partial A}$ representing the importance of each feature map activation pixel for the predicted category $c$.
3. **Weight Pool**: Averages the gradients over spatial dimensions to get channel importance weights $\alpha_c^k$.
4. **Relu Blending**: Computes a weighted sum of activation channels, applying a Rectified Linear Unit ($\text{ReLU}$) to retain only positive gradients that enhance tumor visibility:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_c^k A^k\right)$$

---

## 📈 Research Summary Preview
When running evaluations, the framework outputs classification logs. The automatic markdown report compiled in `reports/research_report.md` documents:
- Performance comparisons between `OncoNetCNN` and `EfficientNet-B0`/`ResNet50`.
- The regularizing effect of Batch Normalization and Dropout layers in MRI classification.
- Detailed anatomical localization heatmaps showing exact coordinate targets for meningiomas, gliomas, and pituitary tumors.
