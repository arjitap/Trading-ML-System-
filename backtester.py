import pandas as pd
from production_model import train_production_xgb

def run_portfolio_backtest(optimal_threshold: float = 0.4093):
    print(f"Starting Dedicated Strategy Backtester with Threshold: {optimal_threshold:.4f}...")
    
    #Pull the trained champion model and the out-of-sample test space
    model, X_test, y_test = train_production_xgb()
    
    # Extract raw prediction probabilities
    probabilities = model.predict_proba(X_test)[:, 1]
    
    
    backtest_df = pd.DataFrame({
        'Actual_Target': y_test,
        'Win_Probability': probabilities
    })
    
    # Apply the optimized confidence shield filter
    backtest_df['Executed_Trade'] = backtest_df['Win_Probability'] >= optimal_threshold
    trades_taken = backtest_df['Executed_Trade'].sum()
    total_setups = len(backtest_df)
    
    if trades_taken == 0:
        print("Backtest Warning: No trades crossed your threshold criteria.")
        return
    
    # Filter down to only trades that were approved by the threshold filter
    executed_set = backtest_df[backtest_df['Executed_Trade'] == True]
    wins = (executed_set['Actual_Target'] == 1).sum()
    losses = (executed_set['Actual_Target'] == 0).sum()
    
    #Calculate Core Financial Metrics
    win_rate = (wins / trades_taken) * 100 if trades_taken > 0 else 0
    
    # Financial assumption based on a standard risk units profile ($100 risk per trade at 2.0 RR)
    gross_profit = wins * 200
    gross_loss = losses * 100
    net_pnl = gross_profit - gross_loss
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print("\n==================== ALGORITHMIC BACKTEST REPORT ====================")
    print(f"Total Technical Crossovers Monitored  : {total_setups}")
    print(f"Trades Approved by XGBoost Cutoff     : {trades_taken} (Filtered out {total_setups - trades_taken} noisy setups)")
    print(f"Simulation Outcome Breakdown          : {wins} Wins / {losses} Losses")
    print(f"Calculated Strategy Win Rate          : {win_rate:.2f}%")
    print(f"Strategy Profit Factor                : {profit_factor:.2f}")
    print(f"Simulated Strategy Net Return (PnL)   : ${net_pnl:,}")
    print("=====================================================================")

if __name__ == "__main__":
    run_portfolio_backtest()