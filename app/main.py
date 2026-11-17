
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

SUBSET = "FD001"
SEQUENCE_LENGTH = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


# Load the saved preprocessing information.
preprocessing = joblib.load(
    MODEL_DIR / f"scaler_{SUBSET}.joblib"
)

SCALER = preprocessing["scaler"]
FEATURES = preprocessing["features"]

# Load the trained LSTM.
checkpoint = torch.load(
    MODEL_DIR / f"lstm_{SUBSET}.pt",
    map_location=DEVICE,
    weights_only=False
)

MODEL = LSTMRegressor(
    input_size=checkpoint["input_size"]
).to(DEVICE)

MODEL.load_state_dict(checkpoint["model_state_dict"])
MODEL.eval()


app = FastAPI(
    title="Turbofan Predictive Maintenance API",
    description=(
        "Predicts remaining useful life (RUL) "
        "of turbofan engines using an LSTM model."
    ),
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    sequence: list[dict[str, float]] = Field(
        ...,
        description=(
            "Exactly 30 chronological rows of raw sensor "
            "and operating-setting measurements."
        )
    )


@app.get("/")
def home():
    return {
        "message": "Turbofan Predictive Maintenance API",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "LSTM",
        "dataset": SUBSET
    }


@app.get("/metadata")
def metadata():
    return {
        "dataset": SUBSET,
        "sequence_length": SEQUENCE_LENGTH,
        "required_features": FEATURES,
        "input_format": "30 chronological rows of raw feature values",
        "prediction_unit": "operational cycles"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    if len(request.sequence) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly {SEQUENCE_LENGTH} rows are required."
        )

    try:
        raw_values = []

        for row_number, row in enumerate(request.sequence):
            missing = [
                feature for feature in FEATURES
                if feature not in row
            ]

            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Row {row_number + 1} is missing "
                        f"features: {missing}"
                    )
                )

            values = [row[feature] for feature in FEATURES]

            if not np.all(np.isfinite(values)):
                raise HTTPException(
                    status_code=422,
                    detail=f"Row {row_number + 1} contains invalid values."
                )

            raw_values.append(values)

        raw_array = np.asarray(raw_values, dtype=np.float32)

        # Apply the exact scaler fitted during preprocessing.
        scaled_values = SCALER.transform(raw_array)

        input_tensor = torch.tensor(
            scaled_values,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction = MODEL(input_tensor).item()

        prediction = max(0.0, float(prediction))

        return {
            "predicted_RUL_cycles": round(prediction, 2),
            "model": "LSTM",
            "dataset": SUBSET
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
