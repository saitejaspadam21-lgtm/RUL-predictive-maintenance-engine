
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

SUBSET = "FD001"
SEQUENCE_LENGTH = 30
RANDOM_SEED = 42


class LSTMRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.regressor(output[:, -1, :]).squeeze(1)


# SHAP's GradientExplainer works with a model returning
# a two-dimensional output: (batch_size, 1).
class SHAPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(1)


def build_training_sequences(df, features):
    sequences = []

    for _, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        values = engine_df[features].to_numpy(dtype=np.float32)

        for end in range(SEQUENCE_LENGTH - 1, len(values)):
            start = end - SEQUENCE_LENGTH + 1
            sequences.append(values[start:end + 1])

    return np.asarray(sequences, dtype=np.float32)


def build_test_sequences(df, features):
    sequences = []

    for _, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        values = engine_df[features].to_numpy(dtype=np.float32)

        if len(values) >= SEQUENCE_LENGTH:
            sequence = values[-SEQUENCE_LENGTH:]
        else:
            padding = np.repeat(
                values[:1],
                SEQUENCE_LENGTH - len(values),
                axis=0
            )
            sequence = np.concatenate([padding, values], axis=0)

        sequences.append(sequence)

    return np.asarray(sequences, dtype=np.float32)


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Explanation device: {device}")

    train_df = pd.read_csv(
        DATA_DIR / f"train_{SUBSET}_processed.csv"
    )
    test_df = pd.read_csv(
        DATA_DIR / f"test_{SUBSET}_processed.csv"
    )

    preprocessing_info = joblib.load(
        MODEL_DIR / f"scaler_{SUBSET}.joblib"
    )
    features = preprocessing_info["features"]

    checkpoint = torch.load(
        MODEL_DIR / f"lstm_{SUBSET}.pt",
        map_location=device
    )

    model = LSTMRegressor(
        input_size=checkpoint["input_size"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Building sensor sequences...")

    train_sequences = build_training_sequences(
        train_df, features
    )
    test_sequences = build_test_sequences(
        test_df, features
    )

    # Use a small representative sample as SHAP background data.
    rng = np.random.default_rng(RANDOM_SEED)
    background_count = min(30, len(train_sequences))
    background_indices = rng.choice(
        len(train_sequences),
        size=background_count,
        replace=False
    )

    background = torch.tensor(
        train_sequences[background_indices],
        dtype=torch.float32,
        device=device
    )

    # Explain predictions for the first 20 test engines.
    explain_count = min(20, len(test_sequences))
    explain_data = torch.tensor(
        test_sequences[:explain_count],
        dtype=torch.float32,
        device=device
    )

    print("Calculating SHAP values...")
    print("This may take a few minutes on a CPU.")

    explainable_model = SHAPModel(model)
    explainer = shap.GradientExplainer(
        explainable_model,
        background
    )

    shap_values = explainer.shap_values(
        explain_data,
        nsamples=50
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    # Remove the single output dimension if SHAP includes it.
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    if shap_values.ndim != 3:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    # Average absolute contribution across engines and time steps.
    importance = np.mean(
        np.abs(shap_values),
        axis=(0, 1)
    )

    importance_df = pd.DataFrame({
        "feature": features,
        "mean_absolute_shap": importance
    }).sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    output_csv = DATA_DIR / f"lstm_shap_importance_{SUBSET}.csv"
    importance_df.to_csv(output_csv, index=False)

    print("\nTOP 10 FEATURES INFLUENCING LSTM PREDICTIONS")
    print(importance_df.head(10).to_string(index=False))

    # Save a feature-importance chart.
    top_features = importance_df.head(15).sort_values(
        "mean_absolute_shap"
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["feature"],
        top_features["mean_absolute_shap"]
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Sensor / feature")
    plt.title("LSTM Feature Importance - FD001")
    plt.tight_layout()

    chart_path = PROJECT_ROOT / "models" / "lstm_shap_importance.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"\nSaved importance data: {output_csv}")
    print(f"Saved chart: {chart_path}")


if __name__ == "__main__":
    main()
