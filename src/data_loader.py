
from pathlib import Path
import pandas as pd


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of NASA C-MAPSS dataset files
DATA_DIR = PROJECT_ROOT / "dataset" / "raw" / "cmapss" / "data"


# Dataset has 26 columns:
# 2 identification columns + 3 operating settings + 21 sensors
COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss_data(subset="FD001"):
    """
    Load NASA C-MAPSS training, testing,
    and actual RUL data for a selected subset.
    """

    train_path = DATA_DIR / f"train_{subset}.txt"
    test_path = DATA_DIR / f"test_{subset}.txt"
    rul_path = DATA_DIR / f"RUL_{subset}.txt"

    # Verify dataset files exist
    for path in [train_path, test_path, rul_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

    # Load training data
    train_df = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    # Load testing data
    test_df = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    # Load actual Remaining Useful Life values
    rul_df = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )

    return train_df, test_df, rul_df


if __name__ == "__main__":

    train_df, test_df, rul_df = load_cmapss_data("FD001")

    print("\n========== NASA C-MAPSS DATASET ==========")

    print("\nTRAINING DATA")
    print(train_df.head())
    print("Training shape:", train_df.shape)

    print("\nTESTING DATA")
    print(test_df.head())
    print("Testing shape:", test_df.shape)

    print("\nACTUAL TEST RUL")
    print(rul_df.head())
    print("RUL shape:", rul_df.shape)

    print("\nDATASET SUMMARY")
    print("Training engines:", train_df["engine_id"].nunique())
    print("Testing engines:", test_df["engine_id"].nunique())
    print("Number of sensors:", 21)

    print("\nMissing values in training data:",
          train_df.isnull().sum().sum())

    print("\nDataset loaded successfully!")
