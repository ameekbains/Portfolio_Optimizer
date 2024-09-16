import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from pypfopt import EfficientFrontier, risk_models, expected_returns

# 1. Data Preparation
def download_data(symbols, start_date, end_date):
    try:
        data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
        if data.isnull().values.any():
            data = data.fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill missing data
        if data.empty:
            raise ValueError("No data returned from Yahoo Finance.")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

def calculate_features(data, symbols):
    returns = data.pct_change().dropna()
    technical_indicators = pd.DataFrame(index=data.index)
    for symbol in symbols:
        technical_indicators[f'{symbol}_MA20'] = data[symbol].rolling(window=20).mean().pct_change().fillna(0)
    
    features = pd.concat([returns, technical_indicators], axis=1).dropna()
    return features

# 2. Predicting Returns
def train_return_models(features, symbols):
    X = features.drop(columns=symbols)
    y = features[symbols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    models = {}
    for symbol in symbols:
        model = LinearRegression()
        model.fit(X_train, y_train[symbol])
        models[symbol] = model
    
    y_pred = pd.DataFrame(index=y_test.index, columns=symbols)
    for symbol, model in models.items():
        y_pred[symbol] = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for Return Predictions: {mse:.4f}")
    return models, y_pred

# 3. Stochastic Volatility Model (Heston Model)
def simulate_heston(n_paths, n_steps, dt, S0, sigma0, kappa, theta, xi, rho):
    paths = np.zeros((n_steps + 1, n_paths))
    vol_paths = np.zeros((n_steps + 1, n_paths))
    
    paths[0] = S0
    vol_paths[0] = sigma0**2
    
    for i in range(1, n_steps + 1):
        z1 = np.random.normal(size=n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=n_paths)
        
        vol_paths[i] = vol_paths[i - 1] + kappa * (theta - vol_paths[i - 1]) * dt + xi * np.sqrt(vol_paths[i - 1]) * np.sqrt(dt) * z1
        paths[i] = paths[i - 1] * np.exp((mu - 0.5 * vol_paths[i]) * dt + np.sqrt(vol_paths[i]) * np.sqrt(dt) * z2)
        
    return paths, vol_paths

def prepare_volatility_data(returns, symbols):
    n_paths = 1000
    n_steps = len(returns)
    dt = 1 / 252  # Daily steps
    S0 = returns.mean()  # Initial price
    sigma0 = returns.std()  # Initial volatility
    kappa = 1.0  # Mean reversion rate
    theta = 0.2**2  # Long-term mean variance
    xi = 0.1  # Volatility of volatility
    rho = -0.7  # Correlation between asset and volatility
    
    simulated_paths, simulated_vol_paths = simulate_heston(n_paths, n_steps, dt, S0, sigma0, kappa, theta, xi, rho)
    
    predicted_volatility_df = pd.DataFrame(simulated_vol_paths[-1].T, columns=returns.columns)
    
    return predicted_volatility_df

# 4. Portfolio Optimization
def optimize_portfolio(predicted_returns, predicted_volatility):
    expected_returns_mean = predicted_returns.mean()
    print("Expected Returns Mean:")
    print(expected_returns_mean)
    
    cov_matrix = pd.DataFrame(np.cov(predicted_volatility.T), index=predicted_volatility.columns, columns=predicted_volatility.columns)
    print("Covariance Matrix:")
    print(cov_matrix)
    
    ef = EfficientFrontier(expected_returns_mean, cov_matrix)
    
    try:
        weights = ef.max_sharpe()
    except ValueError as e:
        print(f"Error during optimization: {e}")
        weights = ef.max_quadratic_utility()  # Alternative
    
    performance = ef.portfolio_performance()
    
    return weights, performance

# 5. Performance Metrics
def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    downside_deviation = np.sqrt((excess_returns[excess_returns < 0]**2).mean())
    return excess_returns.mean() / downside_deviation if downside_deviation != 0 else float('nan')

def calculate_tracking_error(port_returns, benchmark_returns):
    tracking_error = np.sqrt(((port_returns - benchmark_returns)**2).mean())
    return tracking_error

def evaluate_portfolio_performance(returns, benchmark_returns, risk_free_rate=0.0):
    if returns.empty or benchmark_returns.empty:
        return float('nan'), float('nan')
    
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)
    tracking_error = calculate_tracking_error(returns, benchmark_returns)
    return sortino_ratio, tracking_error

# 6. Backtesting
def backtest_with_predictions(data, symbols, weights, look_back=20):
    if len(weights) != len(symbols):
        print("Mismatch between weights and symbols.")
        return pd.Series()

    portfolio_returns = []

    data_reversed = data.iloc[::-1]
    weights_series = pd.Series(weights, index=symbols)

    for i in range(len(data_reversed) - look_back):
        end_idx = i
        start_idx = i + look_back
        
        current_data = data_reversed.iloc[start_idx:end_idx]
        future_data = data_reversed.iloc[end_idx:end_idx + 1]
        
        future_returns = future_data.pct_change().dropna()
        
        if not future_returns.empty and len(future_returns.columns) == len(symbols):
            try:
                port_returns = future_returns.dot(weights_series)
                portfolio_returns.append(port_returns.mean())
            except Exception as e:
                print(f"Error calculating portfolio returns: {e}")
                continue

    if len(portfolio_returns) == 0:
        print("No valid portfolio returns were calculated.")
        return pd.Series()

    portfolio_returns_series = pd.Series(portfolio_returns, index=data_reversed.index[look_back:])
    cumulative_returns = (1 + portfolio_returns_series).cumprod() - 1

    plt.figure(figsize=(12, 8))
    plt.plot(cumulative_returns, label='Backtested Portfolio Returns with ML Predictions')
    plt.title('Backtested Portfolio Performance with Machine Learning Predictions')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return cumulative_returns

# Main Execution
symbols = ['AAPL', 'MSFT', 'WMT', 'GOOGL', 'NVDA']
start_date = '2020-01-01'
end_date = '2024-09-10'

# Download data
data = download_data(symbols, start_date, end_date)
if not data.empty:
    # Calculate features
    features = calculate_features(data, symbols)

    # Train models and predict returns
    return_models, predicted_returns = train_return_models(features, symbols)
    
    # Simulate volatility with Heston model
    volatility_data = data.pct_change().dropna()
    predicted_volatility = prepare_volatility_data(volatility_data, symbols)

    # Optimize portfolio
    weights, performance = optimize_portfolio(predicted_returns, predicted_volatility)
    print(f"Optimized Weights: {weights}")
    print(f"Expected annual return: {performance[0]:.2%}")
    print(f"Annual volatility: {performance[1]:.2%}")
    print(f"Sharpe ratio: {performance[2]:.2f}")

    # Example benchmark data (use actual benchmark data for more accurate results)
    benchmark_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Backtest the portfolio
    cumulative_returns_ml = backtest_with_predictions(data, symbols, weights)
    
    # Evaluate performance
    sortino_ratio, tracking_error = evaluate_portfolio_performance(cumulative_returns_ml, benchmark_returns)
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Tracking Error: {tracking_error:.2%}")
