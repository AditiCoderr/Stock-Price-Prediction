# Stock Price Prediction using LSTM

**LSTM-Based Deep Learning Model**  
**Stock:** RELIANCE.NS (NSE India)  
**Framework:** TensorFlow / Keras  
**Frontend:** Streamlit  
**Dataset:** NIFTY 50 (Kaggle)<br>
**Dataset Link:** https://www.kaggle.com/datasets/ujjwalrastogi0/nifty-50-latest-5-years-of-data

---

## Project Overview

Stock price prediction is one of the most challenging problems in quantitative finance. Markets are influenced by multiple factors such as economic indicators, investor sentiment, geopolitical events, and historical trends. Traditional statistical models often fail to capture the complex, non-linear relationships present in stock data.

This project presents an **end-to-end deep learning solution** using a **Stacked LSTM (Long Short-Term Memory)** network to forecast stock prices of **RELIANCE.NS**. The model is trained on historical data and deployed as an interactive **Streamlit web application** for real-time predictions.

---

## Objective

The goal of this project is to:

- Build a robust **time-series forecasting model**
- Capture **long-term dependencies** in stock data using LSTM
- Engineer meaningful **technical indicators**
- Evaluate model performance using multiple metrics
- Deploy a **real-time interactive web app**

---


### Situation
Stock price forecasting is complex due to:
- High volatility
- Non-linear dependencies
- Influence of external/global factors

Traditional models like ARIMA struggle with such complexity.

---

### Task
Design and implement a system that:
1. Processes historical stock data (2020–2024)
2. Learns temporal patterns using deep learning
3. Provides accurate predictions
4. Deploys as a live application

---

### Action
- Collected NIFTY 50 dataset from Kaggle
- Filtered **RELIANCE stock data**
- Engineered **15 technical indicators**:
  - Moving Averages (20, 50)
  - RSI (14)
  - MACD
  - Bollinger Bands
  - Volatility
  - Momentum
- Built a **Stacked LSTM Model**:
  - 2 LSTM layers (64 → 32 units)
  - Dropout regularization
- Used **60-day sliding window sequences**
- Applied **hyperparameter tuning (Optuna)**
- Evaluated using multiple performance metrics
- Deployed using **Streamlit with live data (yfinance)**

---


##  Working Methodology

Phase 1, phase 2, phase 3, phase 4 is done in Google colab.

**Code written in Google colab:** <br>
https://colab.research.google.com/drive/1LRlejxu12eRSWm5Iv9dhtd0jmAozXz1e?usp=sharing

Steps:
1. First write code in google colab. Train model in Colab
2. Run the download cell. i.e last cell of the colab notebook which I shared.

  ```bash
   # Run this in your Colab to download everything
   from google.colab import files
   files.download('lstm_stock_model.keras')
   files.download('scaler.pkl')
   files.download('close_scaler.pkl')
   files.download('feature_cols.pkl')
```
3. Move them into VS Code project folder
4. Write code for model_utils.py and app.py in VS code
5. Run
   ```bash
   streamlit run app.py ```
in the VS code terminal.

The files you download from google colab are: (These are important files. Hence don't skip the colab part)

| File| What it is|
|------|--------|
|lstm_stock_model.keras | The trained model |
|caler.pkl | Memory of how all features were scaled |
|close_scaler.pkl | Memory of how prices were scaled |
|feature_cols.pkl | List of which features and in what order |


### Phase 1: Dataset Acquisition
- Source: Kaggle (NIFTY 50 dataset)
- Extracted RELIANCE stock data (2020–2024)

### Phase 2: Data Preprocessing
- Handled missing values (forward/backward fill)
- Feature engineering (15 indicators)
- Applied MinMax Scaling
- Created sequences using 60-day window
---

### Phase 3: Evaluation Metrics

| Metric | Meaning |
|------|--------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Percentage Error |
| R² Score | Variance explained |
| Directional Accuracy | Trend prediction accuracy |

---

### Phase 4: Hyperparameter Tuning
- Tool: Optuna (TPE Sampler)
- Trials: 30
- Tuned:
  - LSTM units
  - Dropout rate
  - Learning rate
  - Batch size
  - Window size

---
### Phase 5:  Writing model_util.py and app.py for Deployment
- model_utils.py = what to compute
- app.py = what to show
- Built using **Streamlit**
- streamlit run app.py  <- this one command launches the entire web app
- Features:
  - Live stock data (yfinance)
  - Interactive charts (Plotly)
  - Forecast slider (7–60 days)
  - CSV export

---

##  Project Structure
stock_prediction_app/<br>
│<br>
├── app.py # Streamlit UI & prediction logic<br>
├── model_utils.py # Data processing & helper functions<br>
├── requirements.txt # Dependencies<br>
│<br>
└── models/<br>
├── lstm_stock_model.keras<br>
├── scaler.pkl<br>
├── close_scaler.pkl<br>
└── feature_cols.pkl<br>

##  How to Run Locally

```bash
# Clone repository
git clone https://github.com/AditiCoderr/Stock-Price-Prediction
cd stock-prediction-app

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

**App will open at:** http://localhost:8501  after deployment

**Disclaimer:**
This project is developed for educational purposes only.
Stock market predictions are inherently uncertain and should NOT be used for real financial decisions.
