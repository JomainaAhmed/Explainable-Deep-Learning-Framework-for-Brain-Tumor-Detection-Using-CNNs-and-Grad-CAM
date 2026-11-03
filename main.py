import os
import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import load_config
from src.utils import set_seed, setup_logging, generate_research_report
from src.dataset import (
    CLASSES,
    BrainMRIDataset,
    download_real_dataset,
    generate_synthetic_mri_dataset,
    split_dataset,
    generate_dataset_statistics
)
from src.transforms import get_train_transforms, get_val_test_transforms
from src.model import CerebroNetCNN, OncoNetCNN, get_transfer_model, get_model_summary_string
from src.train import Trainer
from src.evaluate import evaluate_model
from src.gradcam import GradCAM, overlay_heatmap, get_last_conv_layer
from src.visualization import (
    plot_learning_curves,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_gradcam_grid,
    plot_learned_filters,
    plot_feature_maps
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="CEREBRONET Brain Tumor Classification Framework")
    
    # Mode selection
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config YAML file")
    parser.add_argument("--prepare", action="store_true", help="Download/generate raw data, split and run stats")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset generator instead of download")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of synthetic samples per class")
    
    parser.add_argument("--train", action="store_true", help="Train the model specified in the configuration")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model on test set split")
    parser.add_argument("--gradcam", action="store_true", help="Run Grad-CAM explainability on test samples")
    parser.add_argument("--features", action="store_true", help="Visualize convolutional filters and feature activations")
    
    # CLI Overrides for convenience
    parser.add_argument("--model", type=str, default=None, choices=["custom", "resnet50", "efficientnet_b0", "densenet121"], help="Override model name")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific model checkpoint path to load for eval")
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logging(log_dir="logs", log_name="cerebronet.log")
    logger.info("Starting CEREBRONET execution pipeline...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config.train.seed)
    
    # Resolve overrides
    model_name = args.model if args.model else config.model.name
    epochs = args.epochs if args.epochs else config.train.epochs
    batch_size = args.batch_size if args.batch_size else config.train.batch_size
    lr = args.lr if args.lr else config.train.lr
    
    # Update ConfigNode (if values were overridden)
    config.model.name = model_name
    config.train.epochs = epochs
    config.train.batch_size = batch_size
    config.train.lr = lr
    
    # 1. Dataset Preparation Stage
    if args.prepare:
        logger.info("=== STAGE 1: Dataset Preparation ===")
        raw_dir = config.data.raw_dir
        processed_dir = config.data.processed_dir
        
        # Pull data
        success = False
        if not args.synthetic:
            success = download_real_dataset(raw_dir)
            
        if not success:
            logger.info("Falling back to generating realistic synthetic Brain MRI slices...")
            generate_synthetic_mri_dataset(raw_dir, num_samples_per_class=args.num_samples, img_size=config.data.img_size)
            
        # Split folder
        split_dataset(raw_dir, processed_dir, tuple(config.data.split_ratio))
        
        # Generate statistics report
        stats_path = os.path.join(config.logging.report_dir, "dataset_statistics.md")
        generate_dataset_statistics(processed_dir, stats_path)
        logger.info("Dataset preparation stage completed successfully.")
        
    # Set up device
    if config.train.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.train.device)
        
    # 2. Training Stage
    if args.train:
        logger.info("=== STAGE 2: Model Training ===")
        processed_dir = config.data.processed_dir
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Processed dataset directory not found at {processed_dir}. Run with --prepare first.")
            
        # Load datasets
        train_transform = get_train_transforms(img_size=config.data.img_size)
        val_transform = get_val_test_transforms(img_size=config.data.img_size)
        
        train_dataset = BrainMRIDataset(os.path.join(processed_dir, "train"), transform=train_transform)
        val_dataset = BrainMRIDataset(os.path.join(processed_dir, "val"), transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.train.batch_size, 
            shuffle=True, 
            num_workers=config.data.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.train.batch_size, 
            shuffle=False, 
            num_workers=config.data.num_workers
        )
        
        # Instantiate Model
        if model_name == "custom":
            model = CerebroNetCNN(
                num_classes=config.model.num_classes,
                channels=config.model.channels,
                dropout_rate=config.model.dropout,
                use_batch_norm=config.model.use_batch_norm
            )
        else:
            model = get_transfer_model(model_name, num_classes=config.model.num_classes, pretrained=True)
            
        # Log Summary
        logger.info("\n" + get_model_summary_string(model))
        
        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_dir=config.logging.checkpoint_dir,
            log_dir=config.logging.log_dir,
            model_tag=model_name
        )
        history = trainer.fit()
        
        # Plot curves
        curve_path = os.path.join(config.logging.output_dir, "plots", f"{model_name}_learning_curves.png")
        plot_learning_curves(history, curve_path)
        logger.info("Training stage completed successfully.")

    # Resolve model weights path for downstream tasks (eval, gradcam, features)
    checkpoint_path = args.checkpoint
    if (args.evaluate or args.gradcam or args.features) and not checkpoint_path:
        checkpoint_path = os.path.join(config.logging.checkpoint_dir, f"{model_name}_best.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(config.logging.checkpoint_dir, f"{model_name}_latest.pth")
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No saved weights found for model '{model_name}'. Train a model first, or supply --checkpoint."
            )

    # 3. Evaluation Stage
    if args.evaluate:
        logger.info("=== STAGE 3: Model Evaluation ===")
        processed_dir = config.data.processed_dir
        
        # Dataset
        test_transform = get_val_test_transforms(img_size=config.data.img_size)
        test_dataset = BrainMRIDataset(os.path.join(processed_dir, "test"), transform=test_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.train.batch_size, 
            shuffle=False, 
            num_workers=config.data.num_workers
        )
        
        # Load weights
        if model_name == "custom":
            model = CerebroNetCNN(
                num_classes=config.model.num_classes,
                channels=config.model.channels,
                dropout_rate=config.model.dropout,
                use_batch_norm=config.model.use_batch_norm
            )
        else:
            model = get_transfer_model(model_name, num_classes=config.model.num_classes, pretrained=False)
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        metrics, misclassified = evaluate_model(model, test_loader, device, output_dir=config.logging.output_dir)
        
        print("\n==================================================")
        print(f"EVALUATION METRICS SUMMARY (Model: {model_name})")
        print("==================================================")
        print(f"Test Accuracy : {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro F1 Score: {metrics['f1'] * 100:.2f}%")
        print(f"Macro Recall  : {metrics['recall'] * 100:.2f}%")
        print(f"Macro Precision: {metrics['precision'] * 100:.2f}%")
        print(f"One-vs-Rest ROC-AUC: {metrics['roc_auc'] * 100:.2f}%")
        print("==================================================")
        print("\nClassification Report:\n", metrics["raw_report"])
        
        # Plot visual metrics
        plots_dir = os.path.join(config.logging.output_dir, "plots")
        plot_confusion_matrix(metrics["confusion_matrix"], CLASSES, os.path.join(plots_dir, f"{model_name}_confusion_matrix.png"))
        
        # Collect actual target array and prediction arrays for curves
        y_true, y_probs = [], []
        model.eval()
        with torch.no_grad():
            for imgs, lbls, _ in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                y_true.extend(lbls.numpy().tolist())
                y_probs.extend(probs.tolist())
                
        plot_roc_curve(y_true, y_probs, CLASSES, os.path.join(plots_dir, f"{model_name}_roc_curve.png"))
        plot_precision_recall_curve(y_true, y_probs, CLASSES, os.path.join(plots_dir, f"{model_name}_pr_curve.png"))
        
        # Compile a base research report
        report_path = os.path.join(config.logging.report_dir, "research_report.md")
        generate_research_report(
            output_path=report_path,
            dataset_stats={},
            eval_metrics=metrics
        )
        logger.info(f"Evaluation report compiled and saved to {report_path}")

    # 4. Grad-CAM Stage
    if args.gradcam:
        logger.info("=== STAGE 4: Explainability Analysis (Grad-CAM) ===")
        processed_dir = config.data.processed_dir
        
        # Test Loader
        test_transform = get_val_test_transforms(img_size=config.data.img_size)
        test_dataset = BrainMRIDataset(os.path.join(processed_dir, "test"), transform=test_transform)
        
        # Load weights
        if model_name == "custom":
            model = CerebroNetCNN(
                num_classes=config.model.num_classes,
                channels=config.model.channels,
                dropout_rate=config.model.dropout,
                use_batch_norm=config.model.use_batch_norm
            )
        else:
            model = get_transfer_model(model_name, num_classes=config.model.num_classes, pretrained=False)
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        # Get one sample per class from the dataset for Grad-CAM
        selected_samples = {}
        for idx in range(len(test_dataset)):
            _, label, original_img = test_dataset[idx]
            cls_name = CLASSES[label]
            if cls_name not in selected_samples:
                selected_samples[cls_name] = (idx, original_img)
            if len(selected_samples) == len(CLASSES):
                break
                
        # Generate Grad-CAM overlays
        gradcam_out_dir = os.path.join(config.logging.output_dir, "gradcam")
        os.makedirs(gradcam_out_dir, exist_ok=True)
        
        target_layer = get_last_conv_layer(model_name, model)
        
        for cls_name, (dataset_idx, orig_img) in selected_samples.items():
            img_tensor, label, _ = test_dataset[dataset_idx]
            # Add batch dim, load to device
            input_tensor = img_tensor.unsqueeze(0).to(device)
            input_tensor.requires_grad = True
            
            with GradCAM(model, target_layer) as gcam:
                heatmap = gcam.generate_heatmap(input_tensor, target_class=label)
                
            colored_heatmap, blended = overlay_heatmap(heatmap, orig_img, alpha=0.4)
            
            # Predict
            with torch.no_grad():
                out = model(input_tensor)
                probs = torch.softmax(out, dim=1)[0]
                pred_cls = CLASSES[torch.argmax(probs).item()]
                confidence = float(probs[torch.argmax(probs).item()])
                
            save_path = os.path.join(gradcam_out_dir, f"{model_name}_gradcam_{cls_name}.png")
            plot_gradcam_grid(orig_img, pred_cls, confidence, heatmap, blended, save_path)
            
        logger.info(f"Grad-CAM explainability plots successfully saved in {gradcam_out_dir}")

    # 5. Convolution Features Visualization
    if args.features:
        logger.info("=== STAGE 5: Convolution Features Visualization ===")
        processed_dir = config.data.processed_dir
        
        # Test Loader
        test_transform = get_val_test_transforms(img_size=config.data.img_size)
        test_dataset = BrainMRIDataset(os.path.join(processed_dir, "test"), transform=test_transform)
        
        # Load weights
        if model_name == "custom":
            model = CerebroNetCNN(
                num_classes=config.model.num_classes,
                channels=config.model.channels,
                dropout_rate=config.model.dropout,
                use_batch_norm=config.model.use_batch_norm
            )
        else:
            model = get_transfer_model(model_name, num_classes=config.model.num_classes, pretrained=False)
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        # First layer filter plotting
        filter_path = os.path.join(config.logging.output_dir, "features", f"{model_name}_first_layer_filters.png")
        plot_learned_filters(model, filter_path)
        
        # Intermediate activation maps plotting
        # Pull a sample image tensor
        img_tensor, _, _ = test_dataset[0]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        fmap_path = os.path.join(config.logging.output_dir, "features", f"{model_name}_activation_maps.png")
        plot_feature_maps(model, input_tensor, fmap_path)
        logger.info("Convolutional features visualization complete.")

if __name__ == "__main__":
    main()
