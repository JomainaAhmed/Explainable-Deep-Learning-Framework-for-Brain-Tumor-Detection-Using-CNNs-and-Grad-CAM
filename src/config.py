"""
ONCONET Configuration Module.
Handles loading and parsing of project YAML configuration files.
"""

import os
import yaml
from typing import Any, Dict, Union

class ConfigNode:
    """A dictionary wrapper that enables dot-notation access to config options."""
    def __init__(self, data: Dict[str, Any]):
        for key, val in data.items():
            if isinstance(val, dict):
                setattr(self, key, ConfigNode(val))
            else:
                setattr(self, key, val)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfigNode back to standard dictionary."""
        out = {}
        for key, val in self.__dict__.items():
            if isinstance(val, ConfigNode):
                out[key] = val.to_dict()
            else:
                out[key] = val
        return out

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __repr__(self) -> str:
        return str(self.to_dict())


def load_config(config_path: Union[str, os.PathLike]) -> ConfigNode:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ConfigNode object representing the parsed config structure.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        data = {}

    return ConfigNode(data)
