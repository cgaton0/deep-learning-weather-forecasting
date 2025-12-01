import logging

import numpy as np
import pandas as pd

from src.models.build_model import model_cnn_bilstm
from src.models.evaluate import evaluate_model
from src.models.train import train_model
from src.preprocessing.scaling import load_scaler
from src.utils import project_path, ensure_dir

# HIPERPARÁMETROS

WINDOW_SIZE = 72
TARGET_SIZE = 24
UNITS = 32
DROPOUT = 0.1
BATCH_SIZE = 128
EPOCHS = 100
SEED = 42
TARGET_FEATURE = "T_(degC)"

logging.basicConfig(level=logging.INFO)


def load_windows():
    """Carga las ventanas ya procesadas desde data/processed/."""
    base = project_path("data", "processed")

    x_train = np.load(base / "x_train.npy")
    y_train = np.load(base / "y_train.npy")

    x_val = np.load(base / "x_val.npy")
    y_val = np.load(base / "y_val.npy")

    x_test = np.load(base / "x_test.npy")
    y_test = np.load(base / "y_test.npy")

    return x_train, y_train, x_val, y_val, x_test, y_test


def main():
    # Cargar ventanas.
    logging.info("Cargando ventanas...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_windows()

    # Cargar scaler.
    logging.info("Cargando scaler...")
    scaler = load_scaler("data/processed/scaler.pkl")

    train_df = pd.read_parquet("data/processed/train_raw.parquet")

    # Construir modelo.
    logging.info("Creando modelo...")
    input_shape = x_train.shape[1:]  # (window_size, num_features)
    output_size = y_train.shape[1]  # horizonte
    model = model_cnn_bilstm(
        units=UNITS,
        dropout_rate=DROPOUT,
        input_shape=input_shape,
        output_shape=output_size,
    )

    # Entrenar modelo.
    logging.info("Entrenando modelo...")
    model, history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        seed=SEED,
    )

    # Evaluar modelo.
    logging.info("Evaluando modelo...")
    results = evaluate_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        df_columns=train_df.columns,
    )

    print("\n============ MÉTRICAS ============")
    print(f"Test Loss:   {results['test_loss']:.4f}")
    print(f"Test RMSE:   {results['test_rmse']:.4f}")
    print(f"RMSE Global: {results['rmse_global']:.4f}")
    print(f"MAE Global:  {results['mae_global']:.4f}")
    print(f"Corr:        {results['corr_global']:.4f}")

    print("\nRMSE por horizonte:")
    for i, v in enumerate(results["rmse_h"]):
        print(f"  h+{i+1:02d}: {v:.4f}")

    print("\nMAE por horizonte:")
    for i, v in enumerate(results["mae_h"]):
        print(f"  h+{i+1:02d}: {v:.4f}")

    output_dir = project_path("models")
    ensure_dir(output_dir)

    np.save(output_dir / "y_pred_unscaled.npy", results["y_pred_unscaled"])
    np.save(output_dir / "y_test_unscaled.npy", results["y_test_unscaled"])

    logging.info("Pipeline completado y modelo y resultados guardados.")


if __name__ == "__main__":
    main()
