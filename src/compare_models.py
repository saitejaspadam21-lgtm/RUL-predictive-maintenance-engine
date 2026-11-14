
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"

SUBSET = "FD001"

models = {
    "Random Forest": DATA_DIR / f"baseline_predictions_{SUBSET}.csv",
    "LSTM": DATA_DIR / f"lstm_predictions_{SUBSET}.csv"
}

results = []

for model_name, file_path in models.items():
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {file_path}"
        )

    predictions = pd.read_csv(file_path)

    actual = predictions["actual_RUL"]
    predicted = predictions["predicted_RUL"]

    results.append({
        "Model": model_name,
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted)
    })

comparison = pd.DataFrame(results).set_index("Model")

print("\nMODEL PERFORMANCE COMPARISON")
print("============================")
print(comparison.round(3).to_string())

best_model = comparison["RMSE"].idxmin()

print(f"\nBest model based on RMSE: {best_model}")

output_path = DATA_DIR / f"model_comparison_{SUBSET}.csv"
comparison.to_csv(output_path)

print(f"\nComparison saved to: {output_path}")
