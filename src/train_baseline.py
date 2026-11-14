
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline_model(subset="FD001"):

    print(f"\nTraining Random Forest for {subset}")

    # 1. Load processed training and testing data
    train_df = pd.read_csv(
        DATA_DIR / f"train_{subset}_processed.csv"
    )

    test_df = pd.read_csv(
        DATA_DIR / f"test_{subset}_processed.csv"
    )

    # Load actual test RUL labels
    test_rul = pd.read_csv(
        DATA_DIR / f"test_{subset}_targets.csv"
    )["RUL"].to_numpy()

    # 2. Load feature information from saved scaler
    preprocessing_info = joblib.load(
        MODEL_DIR / f"scaler_{subset}.joblib"
    )

    features = preprocessing_info["features"]

    # 3. Prepare training data
    X_train = train_df[features]
    y_train = train_df["RUL"]

    # 4. Prepare testing data
    # Use the last observed cycle for each test engine
    test_last_rows = (
        test_df.sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .sort_values("engine_id")
    )

    X_test = test_last_rows[features]

    if len(X_test) != len(test_rul):
        raise ValueError(
            "Test predictions and actual RUL labels do not match."
        )

    # 5. Build Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # 6. Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # 7. Predict RUL
    predictions = model.predict(X_test)

    # 8. Evaluate model
    rmse = np.sqrt(
        mean_squared_error(test_rul, predictions)
    )

    mae = mean_absolute_error(
        test_rul, predictions
    )

    r2 = r2_score(
        test_rul, predictions
    )

    print("\n========== MODEL RESULTS ==========")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R2  : {r2:.4f}")

    # 9. Save trained model
    model_path = MODEL_DIR / f"random_forest_{subset}.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved at: {model_path}")

    # 10. Save prediction results
    results = pd.DataFrame({
        "engine_id": test_last_rows["engine_id"].to_numpy(),
        "actual_RUL": test_rul,
        "predicted_RUL": predictions
    })

    results_path = DATA_DIR / f"baseline_predictions_{subset}.csv"
    results.to_csv(results_path, index=False)

    print(f"Predictions saved at: {results_path}")

    return model, results


if __name__ == "__main__":
    model, results = train_baseline_model("FD001")

    print("\nFirst 10 predictions:")
    print(results.head(10))

    print("\nBaseline model training completed!")
