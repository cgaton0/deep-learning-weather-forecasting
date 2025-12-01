import logging

logging.basicConfig(level=logging.INFO)


def clean_columns(df):
    """
    Limpia y selecciona las columnas utilizadas del Jena Climate dataset.
    Pasos:
    - Normaliza nombres de columnas(Substituye espacios).
    - Selecciona un subconjunto de las columnas.
    - Elimina valores físicamente impossibles.
    """

    logging.info("Cleaning dataframe...")

    # Normalizar los nombres de las columnas.
    df.columns = [col.replace(" ", "_") for col in df.columns]

    # Seleccionar columnas que se utilizaran en el análisis.
    selected_columns = [
        "p_(mbar)",
        "T_(degC)",
        "rh_(%)",
        "sh_(g/kg)",
        "wv_(m/s)",
        "wd_(deg)",
    ]

    df = df[selected_columns].copy()

    # Eliminar valores erróneos.
    # Velocidad del viento no puede ser negativa.
    invalid_count = (df["wv_(m/s)"] < 0).sum()
    if invalid_count > 0:
        logging.info(f"Fixing {invalid_count} invalid wind-speed values (<0).")
        df.loc[df["wv_(m/s)"] < 0, "wv_(m/s)"] = 0

    # Comprobar si existen valores nulos.
    if df.isna().sum().sum() > 0:
        logging.warning("NaN values detected in the dataset.")

    return df
