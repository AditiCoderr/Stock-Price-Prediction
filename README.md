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
### Results

<img width="515" height="458" alt="image" src="https://github.com/user-attachments/assets/e8780b0c-e3d1-4aed-86fb-2f823070132d" />
<br>
Feature Correlation Matrix: Every cell shows the correlation between two features — values range from +1.0 (perfect positive correlation, dark red) to -1.0 (perfect negative correlation, dark blue). The diagonal is always 1.0 because every feature is perfectly correlated with itself.<br>
The high correlation between Close, MA_20, MA_50 and MA_200 means these features carry redundant information. However for LSTM this is acceptable — the model learns to weight features automatically.<br>

<img width="1040" height="297" alt="image" src="https://github.com/user-attachments/assets/8b7f5243-a59a-4f96-af62-dd615d733ba9" />
<br>
The predicted line (blue dashed) follows the actual price (gray) very closely throughout all 700 training days. The model has learned the training data well — it captures the general trend, the dips around day 300–350, and the rally from day 500 onwards.
<br>
<img width="1038" height="304" alt="image" src="https://github.com/user-attachments/assets/401564ea-4f8e-4999-8962-13e160278ce0" />
<br>This is where the problem is visible. The actual price (blue) moves between ₹1300–₹1600 with sharp ups and downs. The predicted line (orange) is too smooth — it rises slowly from ₹1300 to ₹1450 but completely misses the sharp spike to ₹1600 around day 120.
This tells you the model is underfitting on validation data — it learned the training patterns but struggles with the sharp rally it hasn't seen before. Stock markets are hard to predict.
<br>
<img width="1035" height="298" alt="image" src="https://github.com/user-attachments/assets/46cab97d-ff48-44df-9f03-aa5c9d228316" />
<br>
The predicted line (green dashed) follows the overall downward trend correctly — it captures that RELIANCE was falling from ₹1500 to ₹1200 during this period. However it is smoother than reality — it misses the sharp daily fluctuations and the temporary recovery spike around day 110–120.
<br>
<img width="1115" height="364" alt="image" src="https://github.com/user-attachments/assets/f683bd4e-d001-4e13-b8b5-f94a6c70e87e" />
<br>
The solid black line is the actual RELIANCE closing price during the test period — roughly 150 trading days (about 6 months).
The dashed purple line is what your LSTM model predicted for those same days.
The shaded purple region between them is the prediction error band — the wider it is, the more the model was off on those days.
<br>
<img width="1112" height="314" alt="image" src="https://github.com/user-attachments/assets/11edde03-a008-4ba7-b87e-badbddd40da1" />
<br>

<img width="1108" height="315" alt="image" src="https://github.com/user-attachments/assets/32a471dd-0ec1-487c-9b3c-17b7477d39fc" />
<br>
The predicted line (blue dashed) tracks the actual price (gray) extremely closely across all 700 training days. The shaded error band is very thin throughout — the model learned the training patterns very well. MAPE of 2.73% is excellent — on average the model was only ₹30 off on a stock trading around ₹1000–1300.
<br>
<img width="1109" height="321" alt="image" src="https://github.com/user-attachments/assets/3c8e1213-303c-425b-a961-5b0f33280c39" />
<br>
The gap between actual and predicted is noticeably wider than training. The orange predicted line is consistently below the actual price — the model underestimates the sharp rally to ₹1600. MAPE of 5.55% is still acceptable for stock prediction — under 10% is generally considered good. The error band is wider during the ₹1500–₹1600 peak because this price level was rarely seen during training.
<br>
<img width="1113" height="307" alt="image" src="https://github.com/user-attachments/assets/0967ec26-7886-4450-a526-58dd0318beb6" />
<br>
This is your strongest result. MAPE of 2.99% is excellent — better than validation despite being completely unseen data. The predicted line correctly follows the downtrend from ₹1500 to ₹1200. The error band is widest at the start (days 0–40) and narrows significantly after day 40 — the same cold start pattern seen in the previous chart.
<br>
<img width="847" height="287" alt="image" src="https://github.com/user-attachments/assets/519b5e58-392b-4158-b80e-7ab8abb2f6ef" />
<br>
* Residuals over Time<br>
Residual = Actual price − Predicted price.<br>
The green shaded area (above zero) means the model under-predicted — actual was higher than predicted. The orange/red shaded area (below zero) means the model over-predicted — actual was lower than predicted.
<br>
* Residual Distribution<br>
This histogram shows how frequently each error size occurred.<br>

Mean = ₹13.69 — the model has a slight positive bias, on average under-predicting by ₹13.69 <br>
Std = ₹52.06 — most errors fall within ±₹52 of the actual price <br>
The distribution is slightly right-skewed — more large positive errors than large negative ones, confirming the under-prediction at the start <br>
The bulk of predictions are clustered near zero error
<br>
Actual vs Predicted Scatter (R² = 0.7317) <br>
Each dot is one trading day — x-axis is the actual price, y-axis is what the model predicted. The red dashed line is the perfect prediction line — if the model was perfect, every dot would sit exactly on it. <br>
R² = 0.7317 — the model explains 73.17% of price variance on test data <br>
<img width="1036" height="368" alt="image" src="https://github.com/user-attachments/assets/15cecc3f-3846-49f4-874f-c5f3db03f7e2" />
<br>
 Pie Chart<br>

Correct direction: 65 days (44.5%) <br>
Wrong direction: 81 days (55.5%) <br>
Directional Accuracy = 44.52% <br>


 Direction Comparison Chart<br>
This shows why the accuracy is low. Both the actual and predicted lines are just rapid alternating spikes between Up (+1) and Down (−1) every single day. The model predicts mostly "Up" every day because RELIANCE was in a general uptrend during training — it never learned to confidently predict short-term Down movements.<br>

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/9074cc46-1c7f-4bd5-9bdc-b085f7c5cdc5" />
<br> Val Loss per Trial<br>
The rapid drop from 0.078 to 0.00097 shows TPE sampler is much smarter than random search — it learned quickly which direction to search.<br>


 Hyperparameter Importance<br>
This is the most impressive chart for your presentation. It shows which hyperparameters had the biggest impact on model performance.<br>
<img width="944" height="459" alt="image" src="https://github.com/user-attachments/assets/b465fede-1d77-4cd4-8e3a-608795ff8615" />
<br>
 Landing Page: 
The home screen before clicking Run prediction. Shows the sidebar with RELIANCE.NS selected, date range 2020–2026, forecast slider at 30 days, and all 6 indicators selected. The main area shows what the app does in bullet points.
<br>
<img width="944" height="184" alt="image" src="https://github.com/user-attachments/assets/5abe5e05-65fe-4a46-96a6-80df1621fc99" />
<br>
Results Page (Metric Cards): 
After clicking Run prediction. Shows 4 metric cards:

RMSE ₹72.16 — average error of ₹72 on a ₹1200+ stock — excellent<br>
MAE ₹57.73 — on average predictions are ₹57 off<br>
MAPE 3.93% — under 5% — strong performance<br>
30-day forecast ▼ ₹1,275 — model predicts RELIANCE will fall to ₹1,275 over next 30 days<br>
<img width="944" height="145" alt="image" src="https://github.com/user-attachments/assets/d33fafae-210f-4aa3-916e-e3a0b066e074" />
<br>
Price Chart: 
The interactive Plotly chart showing full RELIANCE history from 2020 to today with all overlays active.<br>
The red dashed vertical line marks "Today". The green curve extending right of it is the 30-day autoregressive forecast heading toward ₹1,275. The black dotted vertical line separates backtest predictions from historical data.<br>
<img width="944" height="184" alt="image" src="https://github.com/user-attachments/assets/d94b142e-51e9-4650-a6b3-ea27858e6cc3" />
<br>
 Volume Subplot: 
Bar chart showing daily trading volume from 2020–2026. Key observation — volume spikes visible in 2020–2021 (COVID period, high volatility trading) then gradually normalises. Jul 2024 shows 11.91 million shares traded. The subplots share the x-axis with the price chart above.
<br>
<img width="944" height="328" alt="image" src="https://github.com/user-attachments/assets/be2d5307-6d8d-444a-9589-307f27ec977d" />
<br>
RSI Subplot: 
RSI(14) oscillating between 20–80 across the full history. The red dotted line at 70 is overbought, green dotted at 30 is oversold. Jul 2024 RSI = 46.85 — neutral zone, neither overbought nor oversold at that point. The RSI correctly shows multiple overbought spikes during the 2024 rally.<br>
<img width="944" height="291" alt="image" src="https://github.com/user-attachments/assets/a2d3f18a-e178-4c88-b663-5aff1b7307e6" />





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
   streamlit run app.py
   ```
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
