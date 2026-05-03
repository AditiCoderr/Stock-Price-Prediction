# to run this file - type in terminal as: streamlit run app.py

from turtle import width

from pandas import col
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pickle
import tensorflow as tf
from datetime import datetime, timedelta

from model_utils import add_features, prepare_input, predict_next_n_days, compute_metrics

# 1. Page config

st.set_page_config(
    page_title = "Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
#  Session state init
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'backtest_export' not in st.session_state:
    st.session_state.backtest_export = None
if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'fig2' not in st.session_state:
    st.session_state.fig2 = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'df' not in st.session_state:         
    st.session_state.df = None

# 2. CSS for custom styling

st.markdown(""" 
            <style>
            .metric-card{
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            border-left: 4px solid #534AB7;
            margin-bottom: 0.5rem;
            }
            
            .metric-label{ font-siz: 12px; color: #888; margin-bottom: 4px;}
            .metric-value { font-size: 22px; font-weight:600; color: #1a1a2e;}
            .good {border-left-color: #1D9E75; }
            .warn { border-left-color: #EF9F27;}
            .bad { border-left-color: #D85A30;}
            
            </style>
            """, unsafe_allow_html=True)


# 3. Load model and artifacts

@st.cache_resource
def load_model_artifacts():
    model = tf.keras.models.load_model('models/lstm_stock_model.keras')
    with open('models/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('models/close_scaler.pkl', 'rb') as f:close_scaler = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:feature_cols = pickle.load(f)
    return model, scaler, close_scaler, feature_cols

model, scaler, close_scaler, feature_cols = load_model_artifacts()


# 4. Sidebar for user input

with st.sidebar:
    st.title(" Configuration")
    st.markdown("---")

    ticker = st.selectbox(
        "Select stock ticker",
        ["RELIANCE.NS"],  # only what your model was trained on
        index=0
    )

    st.markdown("**Historical date range**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=datetime(2020, 1, 1))
    with col2:
        end_date   = st.date_input("To", value=datetime.today())
    
    forecast_days = st.slider(
        "Forecast horizon (days)", min_value=7, max_value=60, value=30, step=7
    )

    show_indicators = st.multiselect(
        "Overlay indicators",
        ["MA 20", "MA 50", "Bollinger Bands", "RSI", "MACD", "Volume"],
        default=["MA 20", "MA 50"]
    )

    run_btn = st.button("Run prediction", use_container_width=True)
    st.markdown("---")
    st.caption("Model: Stacked LSTM . Trained on NIFTY 50")


# 5. Fetch data

@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start = start, end=end, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


# 6. Main

st.title(" Stock Price Prediction - LSTM Model")
st.markdown("Powered by a stacked LSTM trained on NIFTY 50 data . Built with Streamlit")

if run_btn:
    st.session_state.results_ready = False
    raw_df = None

    with st.spinner(f"Fetching {ticker} data..."):
        raw_df = fetch_data(ticker, start_date, end_date)

    if raw_df is None or raw_df.empty:
        st.error("No data found. Check the ticker or date range.")
        st.stop()

    with st.spinner("Engineering features..."):
        df = add_features(raw_df[['Open','High','Low','Close','Volume']].copy())
        if len(df) < 60:
            st.error("Need at least 60 trading days. Extend your date range.")
            st.stop()
        st.session_state.df = df    

    with st.spinner("Running model inference..."):
        backtest_window = 90
        df_backtest     = df.iloc[-(backtest_window + 60):]
        actual_bt, pred_bt = [], []

        for i in range(60, 60 + backtest_window):
            window_df   = df_backtest.iloc[i - 60:i]
            X           = prepare_input(window_df, scaler, feature_cols)
            pred_scaled = model.predict(X, verbose=0)
            pred_price  = close_scaler.inverse_transform(
                              pred_scaled.reshape(-1, 1))[0][0]
            actual_bt.append(float(df_backtest['Close'].iloc[i]))
            pred_bt.append(float(pred_price))

        # define bt_dates here, not inside the chart section
        bt_dates = df_backtest.index[60:60 + backtest_window]

        rmse, mae, mape = compute_metrics(actual_bt, pred_bt)

        future_preds = predict_next_n_days(
            model, df, scaler, close_scaler, feature_cols, n_days=forecast_days
        )
        future_dates = pd.bdate_range(
            df.index[-1], periods=forecast_days + 1
        )[1:]

    forecast_df = pd.DataFrame({
        'Date'           : [d.strftime('%Y-%m-%d') for d in future_dates],
        'Predicted Close': [f"₹{p:,.2f}" for p in future_preds],
        'Upper (+ 5%)'  : [f"₹{p*1.05:,.2f}" for p in future_preds],
        'Lower (- 5%)'  : [f"₹{p*0.95:,.2f}" for p in future_preds],
    })
    backtest_export = pd.DataFrame({
        'Date'     : bt_dates,
        'Actual'   : actual_bt,
        'Predicted': pred_bt
    })

    st.session_state.results_ready   = True
    st.session_state.forecast_df     = forecast_df
    st.session_state.backtest_export = backtest_export
    st.session_state.metrics         = (rmse, mae, mape, future_preds,
                                         future_dates, actual_bt, pred_bt,
                                         bt_dates)
    
if st.session_state.results_ready:
    df = st.session_state.df
    rmse, mae, mape, future_preds, future_dates, \
    actual_bt, pred_bt, bt_dates = st.session_state.metrics

    forecast_df     = st.session_state.forecast_df
    backtest_export = st.session_state.backtest_export

    last_price    = float(future_preds[0])
    forecast_last = float(future_preds[-1])

    # ── Home page link ────────────────────────────────────────
    st.markdown("""
        <a href="/" target="_self" style="
            display: inline-block;
            margin-bottom: 1rem;
            padding: 6px 16px;
            background: transparent;
            border: 0.5px solid #534AB7;
            border-radius: 8px;
            color: #534AB7;
            text-decoration: none;
            font-size: 13px;">
            ← Home
        </a>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────
    st.markdown("### Model performance (last 90-day backtest)")
    c1, c2, c3, c4 = st.columns(4)
    grade = "good" if mape < 5 else "warn" if mape < 10 else "bad"

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">₹{rmse}</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">₹{mae}</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card {grade}">
            <div class="metric-label">MAPE</div>
            <div class="metric-value">{mape}%</div></div>""",
            unsafe_allow_html=True)
    with c4:
        direction = "▲" if forecast_last > last_price else "▼"
        dir_color = "#1D9E75" if forecast_last > last_price else "#D85A30"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{forecast_days}-day forecast</div>
            <div class="metric-value" style="color:{dir_color}">
                {direction} ₹{forecast_last:,.0f}</div></div>""",
            unsafe_allow_html=True)

    # ── Main Plotly chart ─────────────────────────────────────
    st.markdown("### Price chart")

    n_rows = 1 + ("RSI" in show_indicators) + ("MACD" in show_indicators) + ("Volume" in show_indicators)
    row_h  = [3] + [1] * (n_rows - 1)
    fig    = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_h,
        vertical_spacing=0.04
    )

    # Historical close
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'].values.flatten(),
        name='Close price', line=dict(color='#444441', width=1.5)
    ), row=1, col=1)

    # Backtest predicted line
    fig.add_trace(go.Scatter(
        x=bt_dates, y=pred_bt,
        name='Predicted (backtest)',
        line=dict(color='#534AB7', width=1.5, dash='dot')
    ), row=1, col=1)

    # Future forecast
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        name=f'{forecast_days}-day forecast',
        line=dict(color='#1D9E75', width=2, dash='dash'),
        mode='lines+markers',
        marker=dict(size=4)
    ), row=1, col=1)

    # Forecast confidence band (±5%)
    upper = [p * 1.05 for p in future_preds]
    lower = [p * 0.95 for p in future_preds]
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(29,158,117,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='±5% confidence band',
        showlegend=True
    ), row=1, col=1)

    # Vertical line today
    fig.add_vline(
        x=df.index[-1].timestamp() * 1000,
        line_dash='dash', line_color='#D85A30',
        annotation_text='Today', annotation_position='top right'
    )

    # Optional overlays
    curr_row = 2
    if "MA 20" in show_indicators:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_20'].values.flatten(),
            name='MA 20', line=dict(color='#EF9F27', width=1, dash='dot')
        ), row=1, col=1)

    if "MA 50" in show_indicators:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_50'].values.flatten(),
            name='MA 50', line=dict(color='#378ADD', width=1, dash='dot')
        ), row=1, col=1)

    if "Bollinger Bands" in show_indicators:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'].values.flatten(),
            name='BB Upper', line=dict(color='#B4B2A9', width=0.8)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'].values.flatten(),
            name='BB Lower', line=dict(color='#B4B2A9', width=0.8),
            fill='tonexty', fillcolor='rgba(180,178,169,0.08)'
        ), row=1, col=1)

    if "Volume" in show_indicators:
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'].values.flatten(),
            name='Volume', marker_color='rgba(83,74,183,0.3)'
        ), row=curr_row, col=1)
        fig.update_yaxes(title_text='Volume', row=curr_row, col=1)
        curr_row += 1

    if "RSI" in show_indicators:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'].values.flatten(),
            name='RSI', line=dict(color='#993556', width=1)
        ), row=curr_row, col=1)
        fig.add_hline(y=70, line_dash='dot', line_color='#D85A30',
                      row=curr_row, col=1)
        fig.add_hline(y=30, line_dash='dot', line_color='#1D9E75',
                      row=curr_row, col=1)
        fig.update_yaxes(title_text='RSI', row=curr_row, col=1)
        curr_row += 1

    if "MACD" in show_indicators:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'].values.flatten(),
            name='MACD', line=dict(color='#534AB7', width=1)
        ), row=curr_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'].values.flatten(),
            name='Signal', line=dict(color='#EF9F27', width=1)
        ), row=curr_row, col=1)
        fig.update_yaxes(title_text='MACD', row=curr_row, col=1)

    fig.update_layout(
        height=200 + 300 * n_rows,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table ────────────────────────────────────────
    st.markdown("### Forecast table")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    # ── Backtest chart ────────────────────────────────────────
    st.markdown("### Backtest — actual vs predicted (last 90 days)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=bt_dates, y=actual_bt,
        name='Actual', line=dict(color='#444441', width=1.5)
    ))
    fig2.add_trace(go.Scatter(
        x=bt_dates, y=pred_bt,
        name='Predicted', line=dict(color='#534AB7', width=1.5, dash='dot')
    ))
    fig2.add_traces([go.Scatter(
        x=list(bt_dates) + list(bt_dates[::-1]),
        y=[a * 1.02 for a in actual_bt] + [a * 0.98 for a in actual_bt[::-1]],
        fill='toself', fillcolor='rgba(83,74,183,0.06)',
        line=dict(color='rgba(0,0,0,0)'),
        name='±2% band'
    )])
    fig2.update_layout(
        height=350, hovermode='x unified',
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', y=1.05)
    )
    fig2.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig2.update_yaxes(showgrid=True, gridcolor='#f0f0f0', title='Price (INR)')
    st.plotly_chart(fig2, use_container_width=True)

    # ── Download buttons ──────────────────────────────────────
    st.markdown("### Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️  Download forecast CSV",
            data=forecast_df.to_csv(index=False),
            file_name=f"{ticker}_forecast.csv",
            mime='text/csv',
            use_container_width=True
        )
    with col2:
        st.download_button(
            "⬇️  Download backtest CSV",
            data=backtest_export.to_csv(index=False),
            file_name=f"{ticker}_backtest.csv",
            mime='text/csv',
            use_container_width=True
        )

else:
    # Landing state before button is clicked
    st.info("Select a stock and date range in the sidebar, then click **Run prediction**.")
    st.markdown("""
    **What this app does:**
    - Fetches live OHLCV data via Yahoo Finance
    - Engineers 15 technical indicators (MA, RSI, MACD, Bollinger Bands)
    - Runs a trained stacked LSTM model for inference
    - Backtests predictions on the last 90 trading days
    - Forecasts the next 7–60 business days autoregressively
    - Displays interactive charts and tables for analysis
    """)