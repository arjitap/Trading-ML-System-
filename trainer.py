import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

def train_trading_model(data_path: str = "final_labeled_trades.csv"):
    print("Station 4: Executing multi-model benchmark framework...")
    
    df = pd.read_csv(data_path, index_col=0)
    df = df.sort_index()
    
    # Feature Engineering Core
    df['Price_to_EMA9'] = (df['Close'] - df['EMA_9']) / df['Close']
    df['Price_to_EMA20'] = (df['Close'] - df['EMA_20']) / df['Close']
    df['Price_to_EMA50'] = (df['Close'] - df['EMA_50']) / df['Close']
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=24, min_periods=1).mean()
    
    feature_cols = ['Price_to_EMA9', 'Price_to_EMA20', 'Price_to_EMA50', 'RSI', 'ATR', 'Relative_Volume', 'EMA_Diff']
    X = df[feature_cols]
    y = df['Target']
    
    #  Time-Series Split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Clean calculations for class imbalance scaling
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    calculated_scale_weight = neg_count / pos_count
    
    # Model Configurations (With pipeline scaling for Logistic Regression)
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, scale_pos_weight=calculated_scale_weight,
            random_state=42, eval_metric='logloss'
        )
    }
    
    results_records = []
    trained_models = {}
    
    # Train and Evaluate each model side-by-side
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Capture clean raw metrics as numbers
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        rec_win = recall_score(y_test, preds, pos_label=1.0, zero_division=0)
        f1 = f1_score(y_test, preds, pos_label=1.0, zero_division=0)
        
        results_records.append({
            "Model": name,
            "Accuracy": acc,
            "ROC_AUC": auc,
            "Win_Recall": rec_win,
            "F1_Score": f1
        })
        
    #Build, Sort, and Print the Comparison Dataframe
    comparison_df = pd.DataFrame(results_records)
    comparison_df = comparison_df.sort_values("ROC_AUC", ascending=False)
    
    print("\n=================== MODEL COMPARISON TABLE ===================")
    print(comparison_df.round(3).to_string(index=False))
    print("==============================================================")
    
    # Comparative Feature Importance Extraction (Tree Models Only)
    print("\n Feature Importance Analysis Evaluation:")
    
    rf_importance = pd.DataFrame({
        "Feature": feature_cols,
        "RF_Importance": trained_models["Random Forest"].feature_importances_
    })
    
    xgb_importance = pd.DataFrame({
        "Feature": feature_cols,
        "XGB_Importance": trained_models["XGBoost"].feature_importances_
    })
    
    # Merge the individual tracking summaries on the core Feature names
    importance_matrix = pd.merge(rf_importance, xgb_importance, on="Feature").sort_values("XGB_Importance", ascending=False)
    print(importance_matrix.round(4).to_string(index=False))
    print("--------------------------------------------------------------")
    
    return trained_models

if __name__ == "__main__":
    train_trading_model()