import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train_trading_model(data_path: str = "final_labeled_trades.csv"):
    print("Station 4: Loading dataset and engineering relative training features...")
    
    df = pd.read_csv(data_path, index_col=0)
    df = df.sort_index()
    
    # 1. Feature Engineering: Convert raw numbers into relative percentage differences
    df['Price_to_EMA9'] = (df['Close'] - df['EMA_9']) / df['Close']
    df['Price_to_EMA20'] = (df['Close'] - df['EMA_20']) / df['Close']
    df['Price_to_EMA50'] = (df['Close'] - df['EMA_50']) / df['Close']
    
    # Normalize Volume relative to its historical baseline in the dataset
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=24, min_periods=1).mean()
    
    # 2. Select our clean, normalized feature columns
    feature_cols = ['Price_to_EMA9', 'Price_to_EMA20', 'Price_to_EMA50', 'RSI', 'ATR', 'Relative_Volume', 'EMA_Diff']
    
    X = df[feature_cols]
    y = df['Target']
    
    # 3. Chronological Time-Series Split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training Timeline: {X_train.shape[0]} historical setups.")
    print(f"Testing Timeline: {X_test.shape[0]} unseen future setups.")
    
    # 4. Train the Model with strict regularization to block noise memorization
    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=4,            # Kept shallow to prevent overfitting on noise
        min_samples_leaf=5,
        class_weight="balanced", 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate Outcomes
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    
    print("\n================ CALIBRATED MODEL PERFORMANCE ================")
    print(f"Forward Walk Model Accuracy: {accuracy * 100:.2f}%")
    print(f"True ROC-AUC Score: {roc_auc:.4f}")
    print("\nDetailed Performance Metrics (Forward Testing):")
    print(classification_report(y_test, predictions, target_names=['Loss (0)', 'Win (1)'], zero_division=0))
    print("==============================================================")
    
    return model

if __name__ == "__main__":
    train_trading_model()