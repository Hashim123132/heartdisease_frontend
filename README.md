# ❤️ Heart Disease Prediction

A Streamlit web app that predicts the risk of heart disease from clinical inputs using a K-Nearest Neighbors (KNN) classifier.

## Features

- Simple two-column input form (age, blood pressure, cholesterol, ECG, heart rate, etc.)
- One-click prediction with a clear **High Risk** / **Low Risk** result
- Dark themed, centered layout
- Pre-trained model, scaler, and column schema loaded from `joblib` pickles

## Project Structure

```
.
├── app.py                # Streamlit application
├── knn_heart_model.pkl   # Trained KNN classifier
├── heart_scaler.pkl      # Fitted StandardScaler
└── heart_columns.pkl     # Expected feature column order
```

## Requirements

- Python 3.9+
- streamlit
- pandas
- scikit-learn
- joblib

Install dependencies:

```bash
pip install streamlit pandas scikit-learn joblib
```

## Run Locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How It Works

1. User fills in clinical features via sliders and dropdowns.
2. Categorical values are one-hot encoded into the training schema.
3. Missing columns are filled with `0`, then reordered to match `heart_columns.pkl`.
4. Inputs are scaled with `heart_scaler.pkl`.
5. `knn_heart_model.pkl` predicts `1` (high risk) or `0` (low risk).

## Input Features

| Feature | Description |
| --- | --- |
| Age | Patient age (18–100) |
| Sex | M / F |
| Chest Pain Type | ATA, NAP, TA, ASY |
| Resting Blood Pressure | mm Hg |
| Cholesterol | mg/dL |
| Fasting Blood Sugar | `1` if > 120 mg/dL, else `0` |
| Resting ECG | Normal, ST, LVH |
| Max Heart Rate | bpm |
| Exercise-Induced Angina | Y / N |
| Oldpeak | ST depression |
| ST Slope | Up, Flat, Down |

## Disclaimer

This tool is for educational purposes only and is **not** a medical device. Do not use it for clinical decision-making.

## Deployment

Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) by pointing it at `app.py` in this repository.
