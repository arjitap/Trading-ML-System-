# XGBoost Live Two-Way Algorithmic Trading Assistant

An end-to-end, machine-learning-backed quantitative trading system that upgrades traditional moving average technical strategies into a dynamic predictive model. The system detects exponential moving average (EMA 9/20) crossovers across both bullish and bearish market regimes, feeding a 12-dimensional feature space into an optimized XGBoost Classifier. This model predicts in real time whether an active breakout has a statistically viable probability of reaching its volatility-adjusted target or if it represents a high-risk market fakeout.

---

## Technical Problem Statement and Core Solution
Traditional technical indicators, such as the mechanical Exponential Moving Average (EMA) crossover, perform exceptionally well during sustained trending regimes. However, during sideways consolidation or high-noise market phases, they yield frequent "whiplash fakeouts." These are scenarios where a crossover event occurs, but the price immediately reverses, resulting in sequential drawdown.

This project implements a Machine Learning quality-control layer to mitigate this risk. Rather than executing every crossover indiscriminately, an XGBoost Classifier analyzes the underlying market structure—including price velocity, moving average distance spreads, and momentum drift—at the exact millisecond of the crossover event to calculate a probabilistic safety score, filtering out low-confidence structural setups.

---

## Strategy Backtest Performance Analytics
An out-of-sample forward simulation over a historical test window yielded the following performance metrics, demonstrating the efficiency of upgrading a mechanical rule into an intelligent asset-allocation model:

* Total Moving Average Crosses Scanned: 140
* Core XGBoost Triggered Trades: 139 (68 Bullish Long Positions / 71 Bearish Short Positions)
* Strategy Base Win Rate: 39.57%
* Enforced Risk-Reward Architecture: Asymmetric 2.0 Ratio (Target payout: 2.0x ATR = $200; Stop Loss cut: 1.5x ATR = $100)
* System Profit Factor: 1.31 (Generates $1.31 of gross revenue for every $1.00 of gross loss)
* Simulated Strategy Net Return (PnL): +$2,600

---

## Predictive Modeling and Training Methodology

### 1. Directional Labeling Framework (labeling.py)
The backend engine utilizes a dual-track temporal labeling script. It parses historical data to identify discrete structural crossover events:
* Bullish Crossovers (State: 1): Fast EMA 9 crosses above the Slower EMA 20, signaling upward velocity.
* Bearish Crossovers (State: -1): Fast EMA 9 drops below the Slower EMA 20, signaling downward velocity.

Upon registering a crossover, the labeling engine tracks the asset price up to a 24-hour future horizon to observe which volatility barrier is breached first. The barriers are dynamically calculated using the Average True Range (ATR). If the price hits the target boundary first, the observation is labeled a Success (1). If it hits the stop-loss boundary first, it is labeled a Failure (0).

### 2. Feature Engineering Matrix
The Classifier evaluates a 12-dimensional vector capturing the state of the market at the moment of execution:
* Trend Alignment: Crossover direction (1 or -1)
* Structural Spreads: Price_to_EMA9, Price_to_EMA20, Price_to_EMA50, and EMA_Diff
* Line Trajectories: EMA9_Slope and EMA20_Slope
* Momentum Oscillations: RSI, RSI_Change, and RSI_Above_50
* Volatility and Liquidity Spikes: Relative_Volume and ATR_Pct

### 3. Algorithm Selection: XGBoost (production_model.py)
Extreme Gradient Boosting (XGBoost) was selected as the production model due to its resilience against overfitting when processing highly correlated tabular financial indicators. It maps complex, non-linear interactions between technical slopes and volume spikes far more effectively than linear models or shallow decision trees.

### 4. Probability Threshold Optimization (threshold_optimizer.py)
Standard classification tasks use an arbitrary 50% probability threshold. Because this trading framework implements an asymmetric risk-reward profile where wins yield double the monetary value of losses, a grid-search optimizer was executed on out-of-sample data. The mathematically optimal threshold was determined to be 0.0997 (9.97%). This maximizes strategy recall (1.0000), ensuring the model successfully capitalizes on macro-trends while relying on the asymmetric stop-loss framework to absorb minor consolidations.

---

## Software Architecture and Codebase Workflow
To minimize operational friction and prevent running multiple sequential background files during live trading, the system isolates historical data parsing from live inference tracking across the following specific module configurations:

* **data_loader.py (The Data Ingestion Layer)**: Contains the central data ingestion hooks connecting to the external Yahoo Finance API. It standardizes candle download requests based on user parameters like asset tickers and hourly interval spacing.
* **indicators.py (The Mathematical Feature Factory)**: Evaluates raw Open, High, Low, Close vectors to generate smooth trailing tracking markers including Exponential Moving Averages, RSI momentum oscillators, and Average True Range limits.
* **labeling.py (The Historical Supervisor)**: Analyzes long-term history segments to isolate precise historical EMA crossover events, labels them according to their future directional exit performance, and assigns training target targets.
* **main.py (The Central Pipeline Manager)**: Runs as an offline batch processor. It strings together the data loader, metrics generator, and data labeler to output a master historical dataset saved as `final_labeled_trades.csv`.
* **production_model.py (The Brain Architect)**: Transforms raw data columns into structured scale-weighted features, splits indices into historical test splits, and handles internal parameter instantiation for the XGBClassifier.
* **threshold_optimizer.py (The Risk Auditor)**: Iterates over raw model probability distributions across testing slices to calculate the exact decision boundary configuration needed to balance system risk vs return.
* **backtester.py (The Financial Simulator)**: Executes simulated out-of-sample evaluations of model choices to calculate baseline metrics like net returns, position counts, and multi-directional strategy profit factors.
* **app.py (The Live Inference Dashboard)**: Coordinates runtime execution. Upon activation, it silently reads historical CSV structures, fits the model framework in memory, extracts live real-time web candles, translates global server time zones into Indian Standard Time (IST), and presents interactive strategic indicators.

---

## System Installation and Local Deployment

1. Activate the Virtual Environment Sandbox (Windows PowerShell):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
   .\venv\Scripts\Activate.ps1
2. Compile or Update the Historical Dataset File:
    python main.py
3. Launch the Production User Interface Dashboard:
   PowerShell
   streamlit run app.py