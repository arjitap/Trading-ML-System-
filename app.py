import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from production_model import train_production_xgb
from data_loader import fetch_market_data
from indicators import calculate_metrics

# Set up browser layout configuration
st.set_page_config(page_title="Simple Trading AI Hub", layout="wide")

st.title("🤖 Simple XGBoost Trading Assistant")
st.markdown("This AI scans live Bitcoin data every hour, looks for trend crossovers, and predicts if a trade is safe or dangerous.")

@st.cache_resource
def load_trained_model():
    """Trains the AI once in the background using historical data logs."""
    model, _, _ = train_production_xgb()
    return model

try:
    # 1. Load our AI brain
    model = load_trained_model()
    risk_gate = 0.0997  # Our calculated 9.97% minimum gate
    
    st.sidebar.header("⚙️ Settings")
    ticker = st.sidebar.text_input("Crypto Asset ID", value="BTC-USD")
    
    with st.spinner("Fetching live prices from the internet..."):
        # Fetch fresh data candles
        raw_live_data = fetch_market_data(ticker=ticker, period="5d", interval="1h")
        processed_live_data = calculate_metrics(raw_live_data)
    
    # 2. Setup Crossover Detection Sequence
    processed_live_data['Prev_EMA_Diff'] = processed_live_data['EMA_9'].shift(1) - processed_live_data['EMA_20'].shift(1)
    processed_live_data['Curr_EMA_Diff'] = processed_live_data['EMA_9'] - processed_live_data['EMA_20']
    
    processed_live_data['Crossover'] = 0
    processed_live_data.loc[(processed_live_data['Prev_EMA_Diff'] <= 0) & (processed_live_data['Curr_EMA_Diff'] > 0), 'Crossover'] = 1
    processed_live_data.loc[(processed_live_data['Prev_EMA_Diff'] >= 0) & (processed_live_data['Curr_EMA_Diff'] < 0), 'Crossover'] = -1

    # 3. Compute simple features for the AI
    processed_live_data['Price_to_EMA9'] = (processed_live_data['Close'] - processed_live_data['EMA_9']) / processed_live_data['Close']
    processed_live_data['Price_to_EMA20'] = (processed_live_data['Close'] - processed_live_data['EMA_20']) / processed_live_data['Close']
    processed_live_data['Price_to_EMA50'] = (processed_live_data['Close'] - processed_live_data['EMA_50']) / processed_live_data['Close']
    processed_live_data['Relative_Volume'] = processed_live_data['Volume'] / processed_live_data['Volume'].rolling(window=24, min_periods=1).mean()
    processed_live_data['EMA_Gap_Pct'] = (processed_live_data['EMA_9'] - processed_live_data['EMA_20']) / processed_live_data['EMA_20']
    processed_live_data['EMA9_Slope'] = processed_live_data['EMA_9'].diff()
    processed_live_data['EMA20_Slope'] = processed_live_data['EMA_20'].diff()
    processed_live_data['RSI_Change'] = processed_live_data['RSI'].diff()
    processed_live_data['ATR_Pct'] = processed_live_data['ATR'] / processed_live_data['Close']
    processed_live_data['RSI_Above_50'] = (processed_live_data['RSI'] > 50).astype(int)
    processed_live_data['EMA_Diff'] = (processed_live_data['EMA_9'] - processed_live_data['EMA_20']) / processed_live_data['Close']
    
    feature_cols = [
        'Crossover', 'Price_to_EMA9', 'Price_to_EMA20', 'Price_to_EMA50', 'Relative_Volume', 
        'EMA_Diff', 'EMA_Gap_Pct', 'EMA9_Slope', 'EMA20_Slope', 'RSI_Change', 'ATR_Pct', 
        'RSI_Above_50', 'RSI'
    ]
    
    # Isolate current state
    latest_row = processed_live_data.iloc[[-1]]
    live_cross = latest_row['Crossover'].values[0]
    ai_confidence = model.predict_proba(latest_row[feature_cols])[:, 1][0]
    
    # Convert Global UTC Time to Local Indian Standard Time (IST)
    utc_time = pd.to_datetime(processed_live_data.index[-1])
    local_ist_time = utc_time.tz_convert('Asia/Kolkata')
    
    current_price = processed_live_data['Close'].iloc[-1]
    
    # SECTION 1: SIMPLE KPI DASHBOARD
    st.header(f"📊 Current Market State: {ticker}")
    st.markdown(f"**Live Price:** ${current_price:,.2f} | **Last Updated (India Time):** {local_ist_time.strftime('%Y-%m-%d %I:%M %p')}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="AI Trade Safety Confidence Score", value=f"{ai_confidence * 100:.1f}%")
        st.caption("How safe the AI thinks this trade setup is based on history.")
    with col2:
        st.metric(label="Minimum Safety Limit Required", value=f"{risk_gate * 100:.1f}%")
        st.caption("The mathematical cutoff score needed to clear risk filters.")
    with col3:
        st.markdown("**Final Action Recommendation**")
        if ai_confidence >= risk_gate and live_cross != 0:
            st.subheader("🟢 READY TO TRADE")
        else:
            st.subheader("❌ HOLD / DO NOT ENTER")
        st.caption("Whether the strategy parameters line up for an active entry.")

    st.divider()

    # SECTION 2: NEW SIMPLE CROSSOVER NOTIFICATION CENTRE
    st.header("🚨 Live Trend Crossover Detector Alerts")
    
    if live_cross == 1:
        st.success(f"BULLISH CROSSOVER SPOTTED! The fast trend line (9 EMA) just crossed ABOVE the slower line (20 EMA). This indicates upward price speed.")
        if ai_confidence >= risk_gate:
            st.info("👉 **SUGGESTED ACTION:** Open a BUY (Long) trade setup. Set your Take Profit targets higher using your standard ATR channel spacing guidelines.")
        else:
            st.warning("⚠️ **SUGGESTED ACTION:** Even though a buy cross happened, the AI confidence score is too low. This looks like a dangerous fakeout trap. SKIP this trade.")
            
    elif live_cross == -1:
        st.error(f" BEARISH CROSSOVER SPOTTED! The fast trend line (9 EMA) just dropped BELOW the slower line (20 EMA). This indicates downward price pressure.")
        if ai_confidence >= risk_gate:
            st.info("👉 **SUGGESTED ACTION:** Open a SELL/SHORT trade setup to profit as the asset drops. Apply your stop loss rules firmly above the recent price high.")
        else:
            st.warning("⚠️ **SUGGESTED ACTION:** A sell cross happened, but the AI metrics indicate high market noise. Avoid entering a short position right here.")
            
    else:
        st.info("⚪ **No fresh trend crossover happened in the last hour.** The lines are moving parallel or waiting for a breakout direction. Keep monitoring and **STAND BY**.")

except Exception as e:
    st.error(f"Dashboard Display Error: {e}")