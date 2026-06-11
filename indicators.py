import pandas as pd
import pandas_ta as ta

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating technical indicators")
    
    
    processed_df = df.copy()
    
    # Calculate EMAs
    processed_df['EMA_9'] = ta.ema(processed_df['Close'], length=9)
    processed_df['EMA_20'] = ta.ema(processed_df['Close'], length=20)
    processed_df['EMA_50'] = ta.ema(processed_df['Close'], length=50)
    
    # Momentum & Volatility
    processed_df['RSI'] = ta.rsi(processed_df['Close'], length=14)
    processed_df['ATR'] = ta.atr(processed_df['High'], processed_df['Low'], processed_df['Close'], length=14)
    
    #MACD
    macd_df = ta.macd(processed_df['Close'], fast=12, slow=26, signal=9)
    
    # Combine them safely
    processed_df = pd.concat([processed_df, macd_df], axis=1)
    
    # Simple Derived feature
    processed_df['EMA_Diff'] = (processed_df['EMA_9'] - processed_df['EMA_20']) / processed_df['Close']
    
    # Drop empty baseline rows
    processed_df.dropna(inplace=True)
    
    print(f"Success! Added columns. Current rows: {len(processed_df)}")
    return processed_df

if __name__ == "__main__":
    print(" Testing Indicators Module standalone...")
    from data_loader import fetch_market_data
    
    raw_data = fetch_market_data("BTC-USD", period="2mo", interval="1h")
    calculated_data = calculate_metrics(raw_data)
    
    print("\nVerify your new columns are appearing correctly:")
    print(calculated_data[['Close', 'EMA_9', 'EMA_20', 'EMA_50', 'RSI']].head(3))