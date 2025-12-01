import pandas as pd


def downsample_df(df, time):
    """Recibe un dataframe y un string con la frecuencia de remuestreo.
    Devuelve un dataframe con la frecuencia deseada y con interpolación
    temporal para rellenar los valores faltantes.
    Args:
        df (pd.DataFrame): Dataframe original.
        time (str): Frecuencia de remuestreo.
    Returns:
        df_resampled (pd.DataFrame): Dataframe remuestreado e interpolado.
    """
    if time is None:
        return df
    else:
        return df.resample(time).mean().bfill()
