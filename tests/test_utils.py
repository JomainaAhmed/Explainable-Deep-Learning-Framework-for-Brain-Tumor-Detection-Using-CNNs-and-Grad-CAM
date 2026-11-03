import os
import sys
import unittest
import tempfile
from unittest.mock import MagicMock

# Mock torch if not installed in current environment
if "torch" not in sys.modules:
    try:
        import torch
    except ModuleNotFoundError:
        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock
        sys.modules["torch.backends"] = MagicMock()
        sys.modules["torch.backends.cudnn"] = MagicMock()

from src.utils import set_seed, setup_logging, generate_research_report, dataframe_to_markdown
import pandas as pd

class TestUtils(unittest.TestCase):
    def test_set_seed(self):
        # Should execute without errors
        set_seed(42)

    def test_setup_logging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(log_dir=temp_dir, log_name="test.log")
            log_file = os.path.join(temp_dir, "test.log")
            self.assertTrue(os.path.exists(log_file))

    def test_generate_research_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "research_report.md")
            dataset_stats = {"total": 400}
            eval_metrics = {
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.90,
                "f1": 0.905,
                "roc_auc": 0.96,
                "per_class": {
                    "glioma": {"precision": 0.90, "recall": 0.89, "f1": 0.895, "support": 100},
                    "meningioma": {"precision": 0.92, "recall": 0.91, "f1": 0.915, "support": 100},
                    "pituitary": {"precision": 0.93, "recall": 0.92, "f1": 0.925, "support": 100},
                    "no_tumor": {"precision": 0.95, "recall": 0.94, "f1": 0.945, "support": 100}
                }
            }
            ablation_results = {
                "model_d": {"name": "Full Architecture", "val_accuracy": 0.92, "val_f1": 0.91, "val_loss": 0.25}
            }
            transfer_results = {
                "ResNet50": {"test_accuracy": 0.94, "test_f1": 0.935, "total_params": 23000000, "trainable_params": 100000}
            }
            
            generate_research_report(
                output_path=report_path,
                dataset_stats=dataset_stats,
                eval_metrics=eval_metrics,
                ablation_results=ablation_results,
                transfer_results=transfer_results
            )
            
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("CEREBRONET", content)
                self.assertIn("CerebroNetCNN", content)
                self.assertIn("Full Architecture", content)
                self.assertIn("ResNet50", content)

    def test_dataframe_to_markdown(self):
        df = pd.DataFrame({
            "Model": ["CerebroNetCNN", "ResNet50"],
            "Accuracy": [0.91, 0.94]
        })
        md = dataframe_to_markdown(df)
        self.assertIn("CerebroNetCNN", md)
        self.assertIn("Accuracy", md)

if __name__ == "__main__":
    unittest.main()
