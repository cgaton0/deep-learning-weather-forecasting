"""
Configuration utilities for loading YAML experiment files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from src.utils import project_path, ensure_dir

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def load_config(path: PathLike) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.
        Relative paths are resolved from the project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the YAML file is empty.
    """
    path = project_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.info("Loading configuration from: %s", path)

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty configuration file: {path}")

    return config


def save_config(config: Dict[str, Any], path: PathLike) -> Path:
    """
    Save a configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save.
    path : str or Path
        Output path. Relative paths are resolved from the project root.

    Returns
    -------
    Path
        Path to the saved configuration file.
    """
    out_path = project_path(path)
    ensure_dir(out_path.parent)

    logger.info("Saving configuration to: %s", out_path)

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return out_path
