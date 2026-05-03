# 📈 Stock Price Prediction using LSTM

**LSTM-Based Deep Learning Model | Minor Project**  
**Stock:** RELIANCE.NS (NSE India)  
**Framework:** TensorFlow / Keras  
**Frontend:** Streamlit  
**Dataset:** NIFTY 50 (Kaggle)

---

## 🚀 Project Overview

Stock price prediction is one of the most challenging problems in quantitative finance. Markets are influenced by multiple factors such as economic indicators, investor sentiment, geopolitical events, and historical trends. Traditional statistical models often fail to capture the complex, non-linear relationships present in stock data.

This project presents an **end-to-end deep learning solution** using a **Stacked LSTM (Long Short-Term Memory)** network to forecast stock prices of **RELIANCE.NS**. The model is trained on historical data and deployed as an interactive **Streamlit web application** for real-time predictions.

---

## 🎯 Objective

The goal of this project is to:

- Build a robust **time-series forecasting model**
- Capture **long-term dependencies** in stock data using LSTM
- Engineer meaningful **technical indicators**
- Evaluate model performance using multiple metrics
- Deploy a **real-time interactive web app**

---

## 🧠 Approach (STAR Method)

### 🔹 Situation
Stock price forecasting is complex due to:
- High volatility
- Non-linear dependencies
- Influence of external/global factors

Traditional models like ARIMA struggle with such complexity.

---

### 🔹 Task
Design and implement a system that:
1. Processes historical stock data (2020–2024)
2. Learns temporal patterns using deep learning
3. Provides accurate predictions
4. Deploys as a live application

---

### 🔹 Action
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

### 🔹 Result
- ✅ MAPE < 5% (high accuracy)
- ✅ R² Score > 0.90
- ✅ Directional Accuracy > 55%
- ✅ Outperformed XGBoost & Prophet models
- ✅ Live forecasting (7–60 days) via Streamlit app

---

## ⚙️ Working Methodology

### Phase 1: Dataset Acquisition
- Source: Kaggle (NIFTY 50 dataset)
- Extracted RELIANCE stock data (2020–2024)

### Phase 2: Data Preprocessing
- Handled missing values (forward/backward fill)
- Feature engineering (15 indicators)
- Applied MinMax Scaling
- Created sequences using 60-day window

### Phase 3: Model Selection
Compared:
- LSTM ✅ (Selected)
- GRU
- XGBoost
- Prophet

---

### Phase 4: Model Architecture
