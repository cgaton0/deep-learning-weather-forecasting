import numpy as np
import pandas as pd


def create_windows_df(df, window_size, target_size, feature_name="T_(degC)"):
    """Recibe un dataframe, un tamaño de ventana y un tamaño de target.
    Devuelve un dataframe con ventanas de tamaño window_size y un target
    de tamaño target_size.
    Args:
        df (pd.DataFrame): Dataframe original.
        window_size (int): Tamaño de la ventana.
        target_size (int): Tamaño del target.
        feature_name (str): Nombre de la columna objetivo.
    Returns:
        x (np.array): Array con ventanas de tamaño window_size.
        y (np.array): Array con target de tamaño.
    """

    values = df.values
    target_idx = df.columns.get_loc(feature_name)

    n = len(df) - window_size - target_size + 1

    x = np.zeros((n, window_size, df.shape[1]), dtype=np.float32)
    y = np.zeros((n, target_size), dtype=np.float32)

    # Creamos ventanas y targets.
    for i in range(n):
        x[i] = values[i : i + window_size]
        y[i] = values[i + window_size : i + window_size + target_size, target_idx]

    return x, y
