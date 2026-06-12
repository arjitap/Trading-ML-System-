import pandas as pd
import numpy as np

def detect_crossovers_and_label(df: pd.DataFrame, risk_reward: float = 2.0) -> pd.DataFrame:
    print(" Executing Directional Crossover Detection and Target Labeling Engine...")
    
    processed_df = df.copy()
    
    
    # 1 = Bullish Cross (9 EMA goes over 20 EMA), -1 = Bearish Cross (9 EMA drops below 20 EMA)
    processed_df['Prev_EMA_Diff'] = processed_df['EMA_9'].shift(1) - processed_df['EMA_20'].shift(1)
    processed_df['Curr_EMA_Diff'] = processed_df['EMA_9'] - processed_df['EMA_20']
    
    processed_df['Crossover'] = 0
    # Bullish Cross condition
    processed_df.loc[(processed_df['Prev_EMA_Diff'] <= 0) & (processed_df['Curr_EMA_Diff'] > 0), 'Crossover'] = 1
    # Bearish Cross condition
    processed_df.loc[(processed_df['Prev_EMA_Diff'] >= 0) & (processed_df['Curr_EMA_Diff'] < 0), 'Crossover'] = -1
    
    processed_df['Target'] = np.nan
    lookahead = 24  # Max time window to hold a trade (24 hours)
    
    # Scanning loop to label BOTH Bullish and Bearish trade outcomes
    for i in range(len(processed_df) - lookahead):
        current_cross = processed_df['Crossover'].iloc[i]
        
        if current_cross == 0:
            continue
            
        entry_price = processed_df['Close'].iloc[i]
        atr = processed_df['ATR'].iloc[i]
        
        # Calculate brackets based on trade direction
        if current_cross == 1:  # Bullish Long Trade
            take_profit = entry_price + (risk_reward * atr)
            stop_loss = entry_price - (1.5 * atr)
            
            # Look forward in time to see which barrier is breached first
            for j in range(1, lookahead + 1):
                future_close = processed_df['Close'].iloc[i + j]
                if future_close >= take_profit:
                    processed_df.iloc[i, processed_df.columns.get_loc('Target')] = 1.0  # Win
                    break
                if future_close <= stop_loss:
                    processed_df.iloc[i, processed_df.columns.get_loc('Target')] = 0.0  # Loss
                    break
                    
        elif current_cross == -1:  # Bearish Short Trade
            take_profit = entry_price - (risk_reward * atr)
            stop_loss = entry_price + (1.5 * atr)
            
            # Look forward in time to see which barrier is breached first
            for j in range(1, lookahead + 1):
                future_close = processed_df['Close'].iloc[i + j]
                if future_close <= take_profit:
                    processed_df.iloc[i, processed_df.columns.get_loc('Target')] = 1.0  # Win (Price fell to target)
                    break
                if future_close >= stop_loss:
                    processed_df.iloc[i, processed_df.columns.get_loc('Target')] = 0.0  # Loss (Price ripped upward)
                    break

    
    trade_setups = processed_df[processed_df['Crossover'].isin([1, -1])].dropna(subset=['Target'])
    
    print(f"Success! Filtered out quiet periods. Total directional trade setups tracked: {len(trade_setups)}")
    print(trade_setups['Crossover'].value_counts().rename({1: "Bullish Crosses", -1: "Bearish Crosses"}))
    
    return trade_setups