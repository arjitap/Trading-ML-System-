import pandas as pd
import numpy as np

def detect_crossovers_and_label(df: pd.DataFrame, risk_reward: float = 2.0, lookahead: int = 24) -> pd.DataFrame:
    """
    Scans data for EMA 9/20 crossovers and traces future prices to label outcomes.
    """
    print("Scanning for crossovers and calculating trade outcomes...")
    
    
    df['Prev_EMA_9'] = df['EMA_9'].shift(1)
    df['Prev_EMA_20'] = df['EMA_20'].shift(1)
    
    # Initialize a crossover indicator column
    # 1 = Bullish Cross (Buy), -1 = Bearish Cross (Sell), 0 = No Cross
    df['Crossover'] = 0
    
    bullish_cross = (df['Prev_EMA_9'] <= df['Prev_EMA_20']) & (df['EMA_9'] > df['EMA_20'])
    bearish_cross = (df['Prev_EMA_9'] >= df['Prev_EMA_20']) & (df['EMA_9'] < df['EMA_20'])
    
    df.loc[bullish_cross, 'Crossover'] = 1
    df.loc[bearish_cross, 'Crossover'] = -1
    
    #  Horizon Scan: Read future rows to label trades as Win (1) or Loss (0)
    close_prices = df['Close'].values
    atr_values = df['ATR'].values
    crossovers = df['Crossover'].values
    targets = np.full(len(df), np.nan)
    
    
    for i in range(len(df) - lookahead):
        if crossovers[i] == 0:
            continue
            
        entry_price = close_prices[i]
        atr = atr_values[i]
        
        # Set boundaries based on volatility (ATR)
        stop_loss_distance = 1.5 * atr
        take_profit_distance = risk_reward * stop_loss_distance
        
        if crossovers[i] == 1:  # Bullish Buy Entry
            take_profit = entry_price + take_profit_distance
            stop_loss = entry_price - stop_loss_distance
            
            # Scan the subsequent hours
            for j in range(i + 1, i + lookahead):
                if close_prices[j] >= take_profit:
                    targets[i] = 1  # Hit Profit Target first (Win)
                    break
                if close_prices[j] <= stop_loss:
                    targets[i] = 0  # Hit Stop Loss first (Loss)
                    break
                    
        elif crossovers[i] == -1:  # Bearish Sell Entry
            take_profit = entry_price - take_profit_distance
            stop_loss = entry_price + stop_loss_distance
            
            # Scan the subsequent hours
            for j in range(i + 1, i + lookahead):
                if close_prices[j] <= take_profit:
                    targets[i] = 1  # Hit Profit Target first (Win)
                    break
                if close_prices[j] >= stop_loss:
                    targets[i] = 0  # Hit Stop Loss first (Loss)
                    break
                    
    df['Target'] = targets
    
    
    trade_setups = df[df['Crossover'].isin([1, -1])].dropna(subset=['Target'])
    
    print(f" Labeling complete! Extracted {len(trade_setups)} total trade events.")
    return trade_setups

# --- TEST CHECK ---
# This links File 1, File 2, and File 3 together to make sure everything lines up.
if __name__ == "__main__":
    print("Testing Labeling Module ")
    from data_loader import fetch_market_data
    from indicators import calculate_metrics
    
    raw_data = fetch_market_data("BTC-USD", period="3mo", interval="1h")
    calculated_data = calculate_metrics(raw_data)
    labeled_data = detect_crossovers_and_label(calculated_data, risk_reward=2.0)
    
    print("\nOutcome Breakdown for the Machine Learning Model:")
    print(labeled_data['Target'].value_counts().rename({1.0: "Wins (1)", 0.0: "Losses (0)"}))