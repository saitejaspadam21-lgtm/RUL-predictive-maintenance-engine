
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import load_cmapss_data

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Maximum RUL value used for training
RUL_CAP = 125

# Features available in the NASA dataset
FEATURE_COLUMNS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def calculate_training_rul(train_df):
    """Calculate remaining useful life for every training row."""

    train_df = train_df.copy()

    max_cycles = train_df.groupby("engine_id")["cycle"].transform("max")

    train_df["RUL"] = max_cycles - train_df["cycle"]

    # Clip RUL at 125 cycles
    train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CAP)

    return train_df

def calculate_test_rul(test_df, rul_df):
    """Calculate row-level RUL for the test engines."""

    test_df = test_df.copy()

    test_max_cycles = (
        test_df.groupby("engine_id")["cycle"].max()
    )

    # NASA RUL file contains RUL at each engine's final
    # observed test cycle. Match labels by engine ID order.
    test_ids = sorted(test_df["engine_id"].unique())

    if len(test_ids) != len(rul_df):
        raise ValueError(
            "Number of test engines does not match RUL labels."
        )

    final_rul = dict(zip(test_ids, rul_df["RUL"].tolist()))

    test_df["RUL"] = (
    test_df["engine_id"].map(final_rul)
    + test_df["engine_id"].map(test_max_cycles)
    - test_df["cycle"]

    )

    return test_df

def preprocess_data(subset="FD001"):

    print(f"\nPreprocessing {subset}...")

    # 1. Load original data
    train_df, test_df, rul_df = load_cmapss_data(subset)

    # 2. Calculate RUL
    train_df = calculate_training_rul(train_df)
    test_df = calculate_test_rul(test_df, rul_df)

    # 3. Identify constant features using training data only
    feature_columns = FEATURE_COLUMNS.copy()

    constant_columns = [
        column
        for column in feature_columns
        if train_df[column].nunique() <= 1
    ]

    feature_columns = [
        column for column in feature_columns
        if column not in constant_columns
    ]

    print("Constant features removed:", constant_columns)
    print("Features retained:", len(feature_columns))

    # 4. Fit scaler on training data only
    scaler = StandardScaler()

    train_df[feature_columns] = scaler.fit_transform(
        train_df[feature_columns]
    )

    # 5. Apply the same scaler to testing data
    test_df[feature_columns] = scaler.transform(
        test_df[feature_columns]
    )

    # 6. Save processed datasets
    train_output = PROCESSED_DIR / f"train_{subset}_processed.csv"
    test_output = PROCESSED_DIR / f"test_{subset}_processed.csv"
    rul_output = PROCESSED_DIR / f"test_{subset}_targets.csv"
    scaler_output = MODEL_DIR / f"scaler_{subset}.joblib"

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    # Save actual RUL labels for final test evaluation
    rul_df.to_csv(rul_output, index=False)

    # Save scaler and feature information for later predictions
    joblib.dump(
        {
            "scaler": scaler,
            "features": feature_columns,
            "rul_cap": RUL_CAP
        },
        scaler_output
    )

    # 7. Display results
    print("\nPREPROCESSING COMPLETED")
    print("-----------------------")

    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)
    print("Test RUL shape:", rul_df.shape)

    print("\nTraining RUL summary:")
    print(train_df["RUL"].describe())

    print("\nFiles saved:")
    print(train_output)
    print(test_output)
    print(rul_output)
    print(scaler_output)

    return train_df, test_df, rul_df

if __name__ == "__main__":
    train_processed, test_processed, test_targets = (
        preprocess_data("FD001")
    )

    print("\nAll preprocessing tasks completed successfully!")