import pandas as pd


def create_splits_df(df, test_ratio=0.15, val_ratio=0.15):
    """Recibe un dataframe y un ratio de test y validación.
    Devuelve un dataframe para train, val y test.
    Comprueba que los valores de los ratios sean validos.
    Args:
        df (pd.DataFrame): Dataframe original.
        test_ratio (float): Ratio de test.
        val_ratio (float): Ratio de validación.
    Returns:
        train_df (pd.DataFrame): Dataframe para train.
        val_df (pd.DataFrame): Dataframe para validación.
        test_df (pd.DataFrame): Dataframe para test.
    """

    # Validaciones
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio debe estar entre 0 y 1.")

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio debe estar entre 0 y 1.")

    if test_ratio + val_ratio >= 1:
        raise ValueError("La suma de test_ratio y val_ratio debe ser < 1.")

    df = df.copy()

    #  Calcular tamaños
    df_size = len(df)
    test_size = int(df_size * test_ratio)
    val_size = int(df_size * val_ratio)
    train_size = df_size - test_size - val_size

    # Crear splits
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]

    return train_df, val_df, test_df
