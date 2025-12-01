import numpy as np
from matplotlib import pyplot as plt


def plot_history_weather(history):
    """
    Recibe un historial de entrenamiento y muestra las curvas
    de pérdida y de RMSE.
    Args:
        history (tf.keras.callbacks.History): Historial de entrenamiento.
    """

    hist = history.history

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(hist["loss"], label="Train", marker="x")
    ax[0].plot(hist["val_loss"], label="Validation", marker="x")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].legend()

    ax[1].plot(hist["root_mean_squared_error"], label="Train", marker="x")
    ax[1].plot(hist["val_root_mean_squared_error"], label="Validation", marker="x")
    ax[1].set_title("RMSE")
    ax[1].set_xlabel("Epochs")

    fig.suptitle("Weather train history", fontsize=16)
    plt.show()


def plot_prediction(y_true, y_pred, seed):
    """
    Muestra la serie real vs predicha de un horizonte aleatorio.
    """

    np.random.seed(seed)
    n = np.random.randint(0, y_true.shape[1])

    plt.figure(figsize=(10, 5))

    plt.plot(y_true[:, n], label="Real")
    plt.plot(y_pred[:, n], label="Pred")

    plt.xlabel("Horas")
    plt.ylabel("Temperatura")
    plt.title(f"Horizonte {n + 1} horas")
    plt.legend()
    plt.show()


def plot_weather_samples(y_true, y_pred, seed):
    """
    Visualiza 6 muestras aleatorias (3x2).
    Args:
        y_true (np.array): Valores reales.
        y_pred (np.array): Predicciones.
        seed (int): Semilla aleatoria.
    """

    np.random.seed(seed)
    fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(3):
        for j in range(2):
            idx = np.random.randint(0, len(y_true))

            ax[i, j].plot(
                range(1, len(y_true[idx]) + 1), y_true[idx], label="Real", marker="o"
            )
            ax[i, j].plot(
                range(1, len(y_pred[idx]) + 1), y_pred[idx], label="Pred", marker="x"
            )

            ax[i, j].set_title(f"Muestra {idx}")
            if i == 2:
                ax[i, j].set_xlabel("Horas")
            if j == 0:
                ax[i, j].set_ylabel("Temperatura")

    ax[0, 0].legend()
    fig.suptitle("Weather samples", fontsize=16)
    plt.show()


def plot_metric_evolution(metric, metric_name):
    """Recibe una lista con los valores de una metrica por hora y los visualiza."""

    plt.figure(figsize=(10, 5))

    plt.plot(metric, marker="o")

    plt.xticks(range(len(metric)), range(1, len(metric) + 1))
    plt.title(f"{metric_name} por hora")
    plt.xlabel("Horas")
    plt.ylabel(f"{metric_name}")
    plt.show()
