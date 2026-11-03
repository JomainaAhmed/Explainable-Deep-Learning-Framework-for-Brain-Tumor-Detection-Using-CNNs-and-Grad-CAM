import os
import unittest
import tempfile
import yaml
from src.config import load_config, ConfigNode

class TestConfig(unittest.TestCase):
    def test_confignode_attributes(self):
        data = {
            "model": {
                "name": "custom",
                "channels": [32, 64, 128, 256],
                "num_classes": 4
            },
            "train": {
                "lr": 0.001,
                "epochs": 10
            }
        }
        node = ConfigNode(data)
        self.assertEqual(node.model.name, "custom")
        self.assertEqual(node.model.channels, [32, 64, 128, 256])
        self.assertEqual(node.train.lr, 0.001)
        self.assertEqual(node.train.epochs, 10)
        self.assertEqual(node["train"]["epochs"], 10)

        # Test to_dict conversion
        dict_converted = node.to_dict()
        self.assertEqual(dict_converted["model"]["name"], "custom")

    def test_load_base_config(self):
        config_path = "configs/base_config.yaml"
        self.assertTrue(os.path.exists(config_path), f"{config_path} must exist.")
        config = load_config(config_path)
        self.assertEqual(config.model.name, "custom")
        self.assertEqual(config.model.num_classes, 4)
        self.assertEqual(len(config.model.channels), 4)

    def test_load_tuning_config(self):
        config_path = "configs/tuning_config.yaml"
        self.assertTrue(os.path.exists(config_path), f"{config_path} must exist.")
        config = load_config(config_path)
        self.assertIn("learning_rates", config.hyperparameters.__dict__)
        self.assertIn("model_a", config.ablation.__dict__)

if __name__ == "__main__":
    unittest.main()
