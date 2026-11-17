# RUL Predictive Maintenance Engine

A machine learning project that predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines using NASA's C-MAPSS dataset. The project compares a Random Forest baseline with a Long Short-Term Memory (LSTM) neural network and provides a REST API for making predictions.

## Project Overview

Predictive maintenance helps identify when equipment may need maintenance before failure occurs. This project uses historical engine sensor measurements and operating settings to estimate how many cycles an engine can continue operating before reaching its end of useful life.

**Key objectives:**
- Explore and preprocess NASA C-MAPSS engine degradation data.
- Develop a Random Forest regression baseline.
- Build an LSTM model to learn temporal degradation patterns.
- Compare model performance using RMSE, MAE, and R².
- Use SHAP to investigate model feature importance.
- Deploy the trained LSTM model through a FastAPI prediction service.
- Test API functionality using automated pytest tests.

## Dataset

This project uses the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset.

The dataset contains simulated turbofan engine sensor measurements collected over multiple operating cycles.

Four dataset subsets are available:

- FD001
- FD002
- FD003
- FD004

The current model development and evaluation focus on **FD001**.

Dataset files include:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

Dataset source: [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/)

## Project Structure

```text
RUL-predictive-maintenance-engine/
│
├── dataset/
│   ├── raw/
│   │   └── cmapss/
│   │       └── data/
│   └── processed/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── models/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_baseline.py
│   ├── train_lstm.py
│   ├── compare_models.py
│   └── explain_lstm.py
│
├── app/
│   └── main.py
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Technologies Used

- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- PyTorch
- SHAP
- FastAPI
- Uvicorn
- Pytest
- Git and GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/saitejaspadam21-lgtm/RUL-predictive-maintenance-engine.git
cd RUL-predictive-maintenance-engine
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Development

### 1. Data Exploration

Exploratory data analysis examines engine operating cycles, sensor measurements, operating settings, and degradation patterns.

### 2. Data Preprocessing

The preprocessing pipeline:
- Loads the NASA C-MAPSS dataset.
- Calculates training RUL from engine degradation cycles.
- Caps RUL targets at 125 cycles.
- Removes constant sensor features.
- Standardizes features using `StandardScaler`.
- Saves processed datasets and scaler information.

### 3. Random Forest Baseline

A Random Forest regression model provides a baseline for comparison.

### 4. LSTM Model

An LSTM neural network learns temporal patterns from sequences of engine sensor measurements.

The model uses:
- 30-cycle input sequences.
- Two LSTM layers.
- 64 hidden units.
- 20 training epochs.
- Validation-based checkpoint selection.

### 5. Model Explainability

SHAP is used to investigate which input features contribute most to the LSTM model's predictions.

## Model Performance

The following results were obtained on the FD001 test dataset:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Random Forest | 18.135 | 13.135 | 0.810 |
| LSTM | **16.683** | **12.195** | **0.839** |

The LSTM model achieved better performance across all three evaluation metrics.

Compared with the Random Forest baseline, the LSTM reduced RMSE by approximately 8%.

## Running the Project

Run the data preprocessing pipeline:

```bash
python src/preprocessing.py
```

Train the Random Forest baseline:

```bash
python src/train_baseline.py
```

Train the LSTM model:

```bash
python src/train_lstm.py
```

Compare model performance:

```bash
python src/compare_models.py
```

Generate SHAP explanations:

```bash
python src/explain_lstm.py
```

## FastAPI Prediction Service

The trained LSTM model can be accessed through a REST API.

Start the API server from the project root:

```bash
uvicorn app.main:app --reload
```

Open the following URLs:

- API home: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/health
- Model metadata: http://127.0.0.1:8000/metadata
- Interactive API documentation: http://127.0.0.1:8000/docs

The `/predict` endpoint accepts 30 chronological engine measurement records and returns an estimated remaining useful life in cycles.

Use `/metadata` to retrieve the exact feature names expected by the API.

## Running Tests

Run automated API tests:

```bash
python -m pytest tests -v
```

The tests check:
- API availability.
- Health endpoint.
- Model metadata.
- Valid prediction requests.
- Rejection of invalid sequence lengths.

## Important Notes

- The current implementation focuses on the FD001 dataset.
- Predictions are estimates based on simulated NASA engine data.
- The API expects exactly 30 chronological observations.
- Raw datasets and generated model files are excluded from Git tracking through `.gitignore`.
- This project is intended for educational and research purposes, not as a certified aircraft maintenance system.

## Future Improvements

- Extend training and evaluation to FD002, FD003, and FD004.
- Experiment with CNN-LSTM and attention-based architectures.
- Improve model generalization and uncertainty estimation.
- Build a dashboard for monitoring engine health.
- Containerize the API using Docker.
- Add model monitoring and prediction logging.

## Author

**saitejaspadam21**

GitHub: [saitejaspadam21-lgtm](https://github.com/saitejaspadam21-lgtm)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.