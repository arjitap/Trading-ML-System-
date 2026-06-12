import pandas as pd
from xgboost import XGBClassifier

def get_production_data(data_path: str = "final_labeled_trades.csv"):
    """Loads the dataset and prepares the final engineered feature space."""
    df = pd.read_csv(data_path, index_col=0).sort_index()
    
    # Core feature transformations
    df['Price_to_EMA9'] = (df['Close'] - df['EMA_9']) / df['Close']
    df['Price_to_EMA20'] = (df['Close'] - df['EMA_20']) / df['Close']
    df['Price_to_EMA50'] = (df['Close'] - df['EMA_50']) / df['Close']
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=24, min_periods=1).mean()
    df['EMA_Gap_Pct'] = (df['EMA_9'] - df['EMA_20']) / df['EMA_20']
    df['EMA9_Slope'] = df['EMA_9'].diff()
    df['EMA20_Slope'] = df['EMA_20'].diff()
    df['RSI_Change'] = df['RSI'].diff()
    df['ATR_Pct'] = df['ATR'] / df['Close']
    df['RSI_Above_50'] = (df['RSI'] > 50).astype(int)
    df['EMA_Diff'] = (df['EMA_9'] - df['EMA_20']) / df['Close']
    
    feature_cols = [
        'Crossover','Price_to_EMA9', 'Price_to_EMA20', 'Price_to_EMA50', 'Relative_Volume', 'EMA_Diff',
        'EMA_Gap_Pct', 'EMA9_Slope', 'EMA20_Slope', 'RSI_Change', 'ATR_Pct', 'RSI_Above_50', 'RSI'
    ]
    df.dropna(subset=feature_cols + ['Target'], inplace=True)
    
    X = df[feature_cols]
    y = df['Target']
    
    split_index = int(len(df) * 0.8)
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]

def train_production_xgb():
    """Trains the final optimized XGBoost champion model."""
    print("Training Optimized Production XGBoost Champion Model...")
    X_train, X_test, y_train, y_test = get_production_data()
    
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_weight = neg / pos if pos > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model, X_test, y_test

if __name__ == "__main__":
    train_production_xgb()