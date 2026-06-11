import os
import pandas as pd
from data_loader import fetch_market_data
from indicators import calculate_metrics
from labeling import detect_crossovers_and_label

def run_data_pipeline():
    print("Starting Central Data Processing Pipeline Manager...")
    
    # Pipeline Settings
    ticker = "BTC-USD"
    output_filename = "final_labeled_trades.csv"
    
    #  Download raw asset prices
    raw_data = fetch_market_data(ticker=ticker, period="2y", interval="1h")
    
    # Calculate technical indicators and metrics
    data_with_indicators = calculate_metrics(raw_data)
    
    #  Scan for crossovers and label Wins vs Losses
    final_labeled_data = detect_crossovers_and_label(data_with_indicators, risk_reward=2.0)
    
    # Save the complete data architecture to a local CSV file
    print(f"Saving finalized Machine Learning dataset to local disk as '{output_filename}'...")
    final_labeled_data.to_csv(output_filename)
    
    
    if os.path.exists(output_filename):
        print(f"Success! Data pipeline execution complete. File saved safely.")
        print(f"Total structured trade profiles available for training: {len(final_labeled_data)} rows.")
        print("\n--- Current Target Balance (Win/Loss Matrix) ---")
        print(final_labeled_data['Target'].value_counts().rename({1.0: "Wins (1)", 0.0: "Losses (0)"}))
        print("=================================================")
    else:
        print("Error: The pipeline processed the data but failed to write the local CSV file.")

if __name__ == "__main__":
    run_data_pipeline()