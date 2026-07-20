# Deep Learning Weather Forecasting

End-to-end deep learning pipeline for **24-hour multistep temperature forecasting** on the **Jena Climate dataset**. The project transforms raw 10-minute meteorological observations into hourly sequences and trains a **CNN + BiLSTM** model using the previous 72 hours to predict the next 24 temperature values.

## Project overview

This repository implements a modular and reproducible time-series forecasting workflow:

- Automatic download and loading of the Jena Climate dataset
- Feature selection and configurable temporal downsampling
- Chronological train, validation, and test splits
- Leakage-safe standardization fitted on the training split only
- Sliding-window generation for multistep forecasting
- CNN + BiLSTM model training with early stopping and checkpointing
- Global and horizon-level evaluation metrics
- Saved predictions, metrics, model artifacts, and experiment configuration
- Standalone generation of evaluation figures
- YAML-based experiment configuration through the command line

## Dataset

The project uses the [Jena Climate dataset](https://www.bgc-jena.mpg.de/wetter/), collected at the Max Planck Institute for Biogeochemistry in Jena, Germany. It contains meteorological observations recorded every 10 minutes from 2009 to 2016.

The pipeline selects the following variables:

| Feature | Description |
| --- | --- |
| `p_(mbar)` | Atmospheric pressure |
| `T_(degC)` | Temperature — prediction target |
| `rh_(%)` | Relative humidity |
| `sh_(g/kg)` | Specific humidity |
| `wv_(m/s)` | Wind velocity |
| `wd_(deg)` | Wind direction |

The dataset is downloaded automatically the first time the pipeline runs.

## Forecasting setup

The baseline experiment uses:

| Setting | Value |
| --- | --- |
| Input frequency | Hourly |
| Input window | 72 hours |
| Forecast horizon | 24 hours |
| Target | Temperature (`T_(degC)`) |
| Input features | 6 meteorological variables |
| Split strategy | Chronological 70% / 15% / 15% |
| Model | CNN + BiLSTM |

Each sample has shape `(72, 6)`, and the model produces all 24 future temperature values in a single forward pass.

## Pipeline

1. Download and load the raw observations.
2. Clean column names and retain the selected features.
3. Downsample the data using the configured frequency, aggregation method, and missing-value strategy.
4. Create chronological train, validation, and test splits.
5. Fit a `StandardScaler` on the training split and transform all three splits.
6. Generate input and target windows independently for each split.
7. Train the forecasting model and restore its best validation weights.
8. Evaluate predictions in both scaled and original temperature units.
9. Save the model artifacts, training history, metrics, predictions, and resolved run configuration.

Processed arrays are reused on subsequent runs when all required artifacts are already available. If you change a preprocessing or windowing setting, remove or relocate the existing `data/processed/` artifacts before rerunning the experiment so the dataset is rebuilt with the new configuration.

## Model architecture

The network combines local pattern extraction with sequential modeling:

```text
Input (72 × 6)
  → Conv1D (kernel size 5, same padding)
  → Batch normalization
  → ReLU
  → LSTM
  → Bidirectional LSTM
  → Dropout
  → Dense + ReLU
  → Dropout
  → Dense output (24 steps)
```

The baseline uses 32 convolution filters/LSTM units, dropout `0.10`, Adam optimization, and mean squared error loss.

## Project structure

```text
deep-learning-weather-forecasting/
├── configs/
│   └── baseline.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_Jena_Climate_Forecasting_CNN_BiLSTM.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── predictions/
├── scripts/
│   └── generate_figures.py
├── src/
│   ├── data/
│   ├── models/
│   ├── preprocessing/
│   ├── visualizations/
│   ├── config.py
│   └── utils.py
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/cgaton0/deep-learning-weather-forecasting.git
cd deep-learning-weather-forecasting

python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies include TensorFlow/Keras, NumPy, pandas, scikit-learn, PyYAML, Matplotlib, PyArrow, and Joblib.

## Usage

### Run the baseline experiment

```bash
python main.py
```

This is equivalent to:

```bash
python main.py --config configs/baseline.yaml
```

The command downloads and preprocesses the data when necessary, trains the model, evaluates it on the test split, and saves all experiment artifacts.

### Run a custom experiment

Copy the baseline configuration, edit the desired values, and pass the new file to the command line:

```bash
cp configs/baseline.yaml configs/my_experiment.yaml
python main.py --config configs/my_experiment.yaml
```

The YAML configuration groups the experiment settings into the following sections:

- `project`: experiment name and random seed
- `data`: downsampling, aggregation, missing-value handling, and split ratios
- `windowing`: input window, forecast horizon, and target variable
- `model`: architecture metadata and model size
- `training`: batch size, epochs, and callback settings
- `evaluation`: prediction persistence options
- `paths`: processed-data and output locations
- `plots`: visualization defaults

Supported preprocessing options currently include:

- Aggregation: `mean` or `median`
- Missing values: `interpolate`, `ffill`, or `bfill`
- No downsampling: set `downsample_time` to `null`

### Generate evaluation figures

After running an experiment:

```bash
python scripts/generate_figures.py
```

To save and display the figures interactively:

```bash
python scripts/generate_figures.py --show
```

The script generates:

- Training and validation curves
- RMSE and MAE over the forecast horizon
- Predicted-versus-observed comparisons at h+1, h+12, and h+24
- Random 24-hour forecast samples

## Saved artifacts

The default run writes:

```text
data/processed/
├── train_raw.parquet
├── val_raw.parquet
├── test_raw.parquet
├── train_scaled.parquet
├── val_scaled.parquet
├── test_scaled.parquet
├── x_train.npy / y_train.npy
├── x_val.npy   / y_val.npy
└── x_test.npy  / y_test.npy

outputs/
├── models/
│   ├── CNN_BiLSTM.keras
│   └── scaler.joblib
├── metrics/
│   ├── history.json
│   ├── metrics.json
│   └── run_config.yaml
├── predictions/
│   ├── y_pred_unscaled.npy
│   └── y_test_unscaled.npy
└── figures/
    └── *.png
```

Saving the resolved configuration alongside the metrics makes each run easier to reproduce and audit.

## Evaluation

The evaluation pipeline reports global metrics in the original temperature scale and, where applicable, per forecast horizon:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Pearson correlation
- Coefficient of determination (R²)
- Scaled test loss and RMSE

Baseline test results:

| Metric | Value |
| --- | ---: |
| Test loss (scaled) | 0.2074 |
| Test RMSE (scaled) | 0.4555 |
| RMSE | 3.9375 °C |
| MAE | 3.0440 °C |
| Correlation | 0.9059 |
| R² | 0.7447 |

![RMSE over the forecast horizon](outputs/figures/rmse_over_horizon.png)

Performance is strongest at short horizons and gradually degrades as forecast uncertainty increases over the prediction horizon.

## Notebook walkthrough

The notebook [`notebooks/01_Jena_Climate_Forecasting_CNN_BiLSTM.ipynb`](notebooks/01_Jena_Climate_Forecasting_CNN_BiLSTM.ipynb) provides an exploratory walkthrough covering:

- Dataset inspection
- Preprocessing and feature preparation
- Window generation
- Model training
- Evaluation and visualization

The production workflow is implemented in the modular `src/` package and executed through `main.py`.

## Future improvements

- Add persistence and seasonal-naive baselines for a stronger performance reference
- Add unit and integration tests for preprocessing, configuration, and artifact generation
- Validate the full YAML schema and reject inconsistent experiment settings early
- Isolate artifacts by experiment name to simplify multi-run comparisons
- Add automated hyperparameter optimization
- Compare alternative architectures such as TCNs, attention models, and Transformers
- Add experiment tracking and model/version metadata

## License

This project is licensed under the [MIT License](LICENSE).
