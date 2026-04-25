"""
Main entry point for training and evaluating a forecasting model on the Jena Climate dataset.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.config import load_config, save_config
from src.data.build_dataset import build_dataset
from src.models.build_model import model_cnn_bilstm
from src.models.evaluate import evaluate_model
from src.models.train import save_history, train_model
from src.preprocessing.scaling import load_scaler
from src.utils import ensure_dir, project_path, setup_logging

logger = logging.getLogger(__name__)


def load_windows(processed_dir: Path) -> Tuple[np.ndarray, ...]:
    """
    Load windowed train/val/test arrays from disk.

    Parameters
    ----------
    processed_dir : Path
        Directory containing saved numpy arrays.

    Returns
    -------
    Tuple[np.ndarray, ...]
        x_train, y_train, x_val, y_val, x_test, y_test.
    """
    x_train = np.load(processed_dir / "x_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")

    x_val = np.load(processed_dir / "x_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")

    x_test = np.load(processed_dir / "x_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    return x_train, y_train, x_val, y_val, x_test, y_test


def artifacts_exist(processed_dir: Path, scaler_path: Path) -> bool:
    """
    Check whether required dataset artifacts already exist on disk.
    """
    required = [
        processed_dir / "x_train.npy",
        processed_dir / "y_train.npy",
        processed_dir / "x_val.npy",
        processed_dir / "y_val.npy",
        processed_dir / "x_test.npy",
        processed_dir / "y_test.npy",
        processed_dir / "train_raw.parquet",
    ]
    return all(p.exists() for p in required) and scaler_path.exists()


def _save_metrics(results: Dict[str, Any], out_path: Path) -> None:
    """Save evaluation results excluding large arrays to JSON."""
    ensure_dir(out_path.parent)

    serializable: Dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = f"<ndarray shape={value.shape} dtype={value.dtype}>"
        elif isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        else:
            serializable[key] = value

    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Jena Climate forecasting model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_logging(level=logging.INFO)
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    window_cfg = cfg["windowing"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    paths_cfg = cfg["paths"]
    project_cfg = cfg.get("project", {})

    logger.info(
        "Starting experiment: %s",
        project_cfg.get("experiment_name", "unnamed_experiment"),
    )

    processed_dir = project_path(paths_cfg["processed_dir"])
    scaler_path = project_path(paths_cfg["scaler_path"])
    outputs_models_dir = project_path(paths_cfg["models_dir"])
    outputs_metrics_dir = project_path(paths_cfg["metrics_dir"])
    outputs_predictions_dir = project_path(paths_cfg["predictions_dir"])

    ensure_dir(outputs_models_dir)
    ensure_dir(outputs_metrics_dir)
    ensure_dir(outputs_predictions_dir)

    run_config_path = outputs_metrics_dir / "run_config.yaml"
    save_config(cfg, run_config_path)

    if not artifacts_exist(processed_dir, scaler_path):
        logger.info("Processed artifacts not found. Building dataset...")

        x_train, y_train, x_val, y_val, x_test, y_test, scaler = build_dataset(
            downsample_time=data_cfg["downsample_time"],
            aggregation_method=data_cfg["aggregation_method"],
            missing_method=data_cfg["missing_method"],
            test_ratio=data_cfg["test_ratio"],
            val_ratio=data_cfg["val_ratio"],
            window_size=window_cfg["window_size"],
            target_size=window_cfg["target_size"],
            target_feature=window_cfg["target_feature"],
            save=True,
            processed_dir=processed_dir,
            scaler_path=scaler_path,
            reuse_scaler=False,
        )
    else:
        logger.info("Processed artifacts found. Loading from disk...")

        x_train, y_train, x_val, y_val, x_test, y_test = load_windows(processed_dir)
        scaler = load_scaler(scaler_path)

    train_df_path = processed_dir / "train_raw.parquet"
    logger.info("Loading train columns from: %s", train_df_path)
    train_df = pd.read_parquet(train_df_path)

    logger.info("Building model...")
    input_shape = x_train.shape[1:]
    output_size = y_train.shape[1]

    model = model_cnn_bilstm(
        units=model_cfg["units"],
        dropout_rate=model_cfg["dropout"],
        input_shape=input_shape,
        output_size=output_size,
    )

    logger.info("Training model...")
    model, history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=training_cfg["batch_size"],
        epochs=training_cfg["epochs"],
        seed=project_cfg.get("random_seed", training_cfg.get("seed", 42)),
        patience=training_cfg["callbacks"]["early_stopping"]["patience"],
        checkpoint_dir=outputs_models_dir,
    )

    history_path = outputs_metrics_dir / "history.json"
    save_history(history, history_path)

    logger.info("Evaluating model...")
    results = evaluate_model(
        model=model,
        x_test=x_test,
        y_test_scaled=y_test,
        scaler=scaler,
        df_columns=train_df.columns,
        target_feature=window_cfg["target_feature"],
    )

    metrics_path = outputs_metrics_dir / "metrics.json"
    _save_metrics(results, metrics_path)

    if cfg.get("evaluation", {}).get("save_predictions", True):
        np.save(
            outputs_predictions_dir / "y_pred_unscaled.npy",
            results["y_pred_unscaled"],
        )
        np.save(
            outputs_predictions_dir / "y_test_unscaled.npy",
            results["y_test_unscaled"],
        )

    logger.info("Saved metrics to: %s", metrics_path)
    logger.info("Saved predictions to: %s", outputs_predictions_dir)

    logger.info(
        "Test summary | loss=%.4f rmse_scaled=%.4f rmse=%.4f mae=%.4f corr=%.4f r2=%.4f",
        results["test_loss"],
        results["test_rmse"],
        results["rmse_global"],
        results["mae_global"],
        results["corr_global"],
        results["r2_global"],
    )


if __name__ == "__main__":
    main()
