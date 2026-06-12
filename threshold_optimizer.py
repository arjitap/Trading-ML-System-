import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from production_model import train_production_xgb

def optimize_trade_threshold():
    print("Starting Probability Threshold Optimization Suite...")
    
    #  Pull the trained model 
    model, X_test, y_test = train_production_xgb()
    
    # Extract the raw winning probabilities (Column 1)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Calculate precisions and recalls for all possible probability thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)
    
    #  Calculate F1-Scores across all thresholds 
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Locate the exact index where the F1-Score maxes out
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print("\n================ THRESHOLD OPTIMIZATION REPORT ================")
    print(f" Mathematically Optimal Threshold : {best_threshold:.4f}")
    print(f"Maximum Achievable F1-Score      : {best_f1:.4f}")
    print(f"Corresponding Precision Target   : {precisions[best_idx]:.4f}")
    print(f"Corresponding Recall Target      : {recalls[best_idx]:.4f}")
    print("=================================================================")
    
    return best_threshold

if __name__ == "__main__":
    optimize_trade_threshold()