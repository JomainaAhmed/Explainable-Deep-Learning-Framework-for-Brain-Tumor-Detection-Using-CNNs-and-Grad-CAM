import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

from src.metrics import calculate_metrics
from src.dataset import CLASSES

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility to halt training if validation loss plateaus."""
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.praise_limit():
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

    def praise_limit(self) -> int:
        return self.patience


def get_optimizer(model: nn.Module, opt_name: str, lr: float, weight_decay: float) -> optim.Optimizer:
    """Instantiates the specified optimizer."""
    opt_name = opt_name.lower()
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}. Choose 'adam', 'adamw', or 'sgd'.")


def get_scheduler(optimizer: optim.Optimizer, sched_name: str) -> Optional[Any]:
    """Instantiates learning rate scheduler."""
    sched_name = sched_name.lower()
    if sched_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    elif sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    return None


class Trainer:
    """Training runner encapsulating PyTorch model training and validation."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs",
        model_tag: str = "custom"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.model_tag = model_tag
        
        # Directory setup
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up device
        if config.train.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.train.device)
            
        self.model = self.model.to(self.device)
        logger.info(f"Trainer device configured as: {self.device}")

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer and Scheduler
        self.optimizer = get_optimizer(
            self.model, 
            config.train.optimizer, 
            config.train.lr, 
            config.train.weight_decay
        )
        self.scheduler = get_scheduler(self.optimizer, config.train.scheduler)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.train.early_stopping_patience)
        
        # Mixed Precision Scaler
        # AMP is only active on CUDA devices
        self.use_amp = config.train.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, model_tag))
        
        self.start_epoch = 1
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Trains the model for one epoch. Returns (avg_loss, avg_acc)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for images, labels, _ in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero grads
            self.optimizer.zero_grad()
            
            # Forward with Mixed Precision Autocast
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            # Backward and scale
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Stats
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=correct/total)
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, Dict[str, Any]]:
        """Evaluates the model on validation dataset. Returns (val_loss, metrics_dict)."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                running_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                
        val_loss = running_loss / len(self.val_loader.dataset)
        metrics = calculate_metrics(all_labels, all_preds, all_probs, CLASSES)
        return val_loss, metrics

    def fit(self) -> Dict[str, Any]:
        """Runs the complete training and validation cycle."""
        logger.info(f"Starting training on {self.model_tag} for {self.config.train.epochs} epochs.")
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": []
        }
        
        for epoch in range(self.start_epoch, self.config.train.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate()
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1"]
            
            # LR scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Logs
            logger.info(
                f"Epoch {epoch:02d} Summary - "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%"
            )
            
            # TensorBoard logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
            self.writer.add_scalar("F1/Val", val_f1, epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("LearningRate", current_lr, epoch)
            
            # Save history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            
            # Checkpoint saving
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"--> Saved new best model checkpoint with validation loss {val_loss:.4f}")
                
            # Always save latest
            self.save_checkpoint(epoch, is_best=False)
            
            # Early Stopping check
            if self.early_stopping(val_loss):
                logger.warning(f"Early stopping triggered. Halting training at epoch {epoch}.")
                break
                
        self.writer.close()
        return history

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Saves a model checkpoint dictionary to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config
        }
        
        suffix = "best" if is_best else "latest"
        save_path = os.path.join(self.checkpoint_dir, f"{self.model_tag}_{suffix}.pth")
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Resumes model state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Loaded checkpoint at epoch {checkpoint['epoch']} (best val loss: {self.best_val_loss:.4f})")
