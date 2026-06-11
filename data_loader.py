import yfinance as yf
import pandas as pd

def fetch_market_data(ticker: str, period: str = "3mo", interval: str = "1h"):
    
    #Connects to Yahoo Finance API and downloads historical market data.
    
    print(f" Fetching data for {ticker} (Period: {period}, Interval: {interval})...")
    
    # Download data
    raw_data = yf.download(tickers=ticker, period=period, interval=interval)
    
    # Handle multi-index column headers if yfinance formats them that way
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
    raw_data.columns = [str(col) for col in raw_data.columns]    
    # Drop empty rows 
    raw_data.dropna(inplace=True)
    
    print(f"Successfully downloaded {len(raw_data)} data rows!")
    return raw_data


if __name__ == "__main__":
    test_data = fetch_market_data("BTC-USD", period="1mo", interval="1h")
    print("\nSneak peek of the downloaded table:")
    print(test_data.head(3))