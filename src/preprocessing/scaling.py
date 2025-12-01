import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import project_path


def scaler_minmax_df(train_df, val_df, test_df):
    """Recibe un dataframe para train, val y test.
    Devuelve un dataframe escalado con MinMaxScaler y los escaladores.
    Args:
        train_df (pd.DataFrame): Dataframe para train.
        val_df (pd.DataFrame): Dataframe para validación.
        test_df (pd.DataFrame): Dataframe para test.
    Returns:
        scaled_train_df (pd.DataFrame): Dataframe para train escalado.
        scaled_val_df (pd.DataFrame): Dataframe para validación
        scaled_test_df (pd.DataFrame): Dataframe para test escalado.
        scaler (MinMaxScaler): Escalador.
    """

    # Creamos el escalador.
    scaler = StandardScaler()

    # Escalamos las variables.
    scaled_train_df = pd.DataFrame(
        scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index
    )
    scaled_val_df = pd.DataFrame(
        scaler.transform(val_df), columns=val_df.columns, index=val_df.index
    )
    scaled_test_df = pd.DataFrame(
        scaler.transform(test_df), columns=test_df.columns, index=test_df.index
    )

    return scaled_train_df, scaled_val_df, scaled_test_df, scaler


def inverse_scaler(data_scaled, scaler, df_columns, feature_name="T_(degC)"):
    """Recibe un array escalado, un escalador y una lista con los nombres de
    las columnas.
    Devuelve un array desescalado.
    Args:
        data_scaled (np.array): Array escalado.
        scaler (MinMaxScaler): Escalador.
        df_columns (list): Lista con los nombres de las columnas.
        feature_name (string): Variable a desescalar.
    Returns:
        data_rescaled (np.array): Array desescalado.
    """

    # Índice de la variable que desescalamos
    temp_idx = df_columns.get_loc(feature_name)
    N, horizon = data_scaled.shape

    data_rescaled = np.zeros_like(data_scaled)

    # Para cada horizonte, se construye un dummy y se desescala
    for h in range(horizon):
        dummy = np.zeros((N, len(df_columns)))
        dummy[:, temp_idx] = data_scaled[:, h]

        inv = scaler.inverse_transform(dummy)
        data_rescaled[:, h] = inv[:, temp_idx]

    return data_rescaled


def save_scaler(scaler, path):
    """
    Guarda un escalador en disco usando joblib.
    """
    path = project_path(path)

    joblib.dump(scaler, path)


def load_scaler(path):
    """
    Carga un escalador previamente guardado.
    """

    path = project_path(path)

    scaler = joblib.load(path)
    return scaler
