
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SUBSET = "FD001"
SEQUENCE_LENGTH = 30
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def make_sequences(df, features, sequence_length):
    sequences = []
    targets = []

    for _, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        values = engine_df[features].to_numpy(dtype=np.float32)
        rul_values = engine_df["RUL"].to_numpy(dtype=np.float32)

        for end in range(sequence_length - 1, len(values)):
            start = end - sequence_length + 1
            sequences.append(values[start:end + 1])
            targets.append(rul_values[end])

        # Pad short engine histories by repeating their first row.
        if len(values) < sequence_length:
            padding = np.repeat(
                values[:1],
                sequence_length - len(values),
                axis=0
            )
            sequence = np.concatenate([padding, values], axis=0)
            sequences.append(sequence)
            targets.append(rul_values[-1])

    return np.asarray(sequences), np.asarray(targets)


def make_test_sequences(df, features, sequence_length):
    sequences = []
    engine_ids = []

    for engine_id, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        values = engine_df[features].to_numpy(dtype=np.float32)

        if len(values) >= sequence_length:
            sequence = values[-sequence_length:]
        else:
            padding = np.repeat(
                values[:1],
                sequence_length - len(values),
                axis=0
            )
            sequence = np.concatenate([padding, values], axis=0)

        sequences.append(sequence)
        engine_ids.append(engine_id)

    return np.asarray(sequences), np.asarray(engine_ids)


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
        last_output = output[:, -1, :]
        return self.regressor(last_output).squeeze(1)


def train_lstm(subset=SUBSET):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training device: {device}")

    train_df = pd.read_csv(
        DATA_DIR / f"train_{subset}_processed.csv"
    )
    test_df = pd.read_csv(
        DATA_DIR / f"test_{subset}_processed.csv"
    )
    test_rul = pd.read_csv(
        DATA_DIR / f"test_{subset}_targets.csv"
    )["RUL"].to_numpy()

    preprocessing_info = joblib.load(
        MODEL_DIR / f"scaler_{subset}.joblib"
    )
    features = preprocessing_info["features"]

    # Split by engine, not by individual rows, to avoid leakage.
    engine_ids = np.sort(train_df["engine_id"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_ids = rng.permutation(engine_ids)

    split_index = int(len(shuffled_ids) * 0.8)
    train_ids = set(shuffled_ids[:split_index])
    val_ids = set(shuffled_ids[split_index:])

    train_part = train_df[train_df["engine_id"].isin(train_ids)]
    val_part = train_df[train_df["engine_id"].isin(val_ids)]

    X_train, y_train = make_sequences(
        train_part, features, SEQUENCE_LENGTH
    )
    X_val, y_val = make_sequences(
        val_part, features, SEQUENCE_LENGTH
    )

    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    X_val_tensor = torch.tensor(X_val).to(device)
    y_val_tensor = torch.tensor(y_val).to(device)

    model = LSTMRegressor(len(features)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_X)

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(
                val_predictions, y_val_tensor
            ).item()

        average_train_loss = total_loss / len(train_dataset)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {average_train_loss:.3f} | "
            f"Validation loss: {val_loss:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    # Predict RUL at the final observed cycle of each test engine.
    X_test, test_engine_ids = make_test_sequences(
        test_df, features, SEQUENCE_LENGTH
    )

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test).to(device)
        predictions = model(test_tensor).cpu().numpy()

    predictions = np.maximum(predictions, 0)

    if len(predictions) != len(test_rul):
        raise ValueError(
            "Number of predictions does not match test RUL labels."
        )

    rmse = np.sqrt(mean_squared_error(test_rul, predictions))
    mae = mean_absolute_error(test_rul, predictions)
    r2 = r2_score(test_rul, predictions)

    print("\nLSTM test results")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.3f}")

    model_path = MODEL_DIR / f"lstm_{subset}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(features),
            "sequence_length": SEQUENCE_LENGTH,
            "features": features
        },
        model_path
    )

    results = pd.DataFrame({
        "engine_id": test_engine_ids,
        "actual_RUL": test_rul,
        "predicted_RUL": predictions
    })

    results.to_csv(
        DATA_DIR / f"lstm_predictions_{subset}.csv",
        index=False
    )

    print(f"\nSaved model: {model_path}")
    print("\nFirst 10 predictions:")
    print(results.head(10).to_string(index=False))

    return model, results


if __name__ == "__main__":
    train_lstm()
