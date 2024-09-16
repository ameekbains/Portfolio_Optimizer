
## Financial Analysis and Portfolio Optimization

This repository contains a Python-based financial analysis toolkit designed to download stock data, predict returns, simulate volatility using the Heston model, optimize portfolios, and backtest performance. It leverages several popular libraries and techniques to provide a comprehensive solution for analyzing and managing investment portfolios.

### Features

- **Data Preparation**: 
  - Download historical stock data using `yfinance`.
  - Handle missing data with forward and backward filling.
  
- **Feature Engineering**: 
  - Calculate technical indicators such as moving averages.
  - Compute percentage returns for use in predictive models.

- **Return Prediction**: 
  - Train Linear Regression models to predict future stock returns based on historical data and technical indicators.
  - Evaluate model performance using Mean Squared Error.

- **Volatility Simulation**: 
  - Simulate future price paths and volatility using the Heston Stochastic Volatility Model.
  - Generate simulated volatility data for portfolio optimization.

- **Portfolio Optimization**: 
  - Use the `pypfopt` library to optimize portfolio weights for maximum Sharpe ratio.
  - Analyze portfolio performance with expected returns, volatility, and Sharpe ratio.

- **Performance Metrics**: 
  - Calculate Sortino Ratio and Tracking Error to assess portfolio performance.
  - Compare portfolio performance against a benchmark.

- **Backtesting**: 
  - Backtest portfolio strategies using historical data and visualize performance with cumulative returns plots.

### Requirements

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `scikit-learn`
- `tensorflow`
- `pypfopt`

### Installation

To get started, clone the repository and install the required packages using pip:

```bash
git clone https://github.com/your-username/financial-analysis.git
cd financial-analysis
pip install -r requirements.txt
```

### Usage

1. **Configure Symbols and Date Range**: Modify the `symbols` list and `start_date`, `end_date` variables to specify the stocks and time period for analysis.
2. **Run the Script**: Execute the script to download data, train models, simulate volatility, optimize the portfolio, and backtest performance.

### Example

Here's a brief example of how to use the toolkit:

```python
# Import necessary libraries and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from pypfopt import EfficientFrontier

# Define your stock symbols and date range
symbols = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2020-01-01'
end_date = '2024-01-01'

# Download data and perform analysis
data = download_data(symbols, start_date, end_date)
features = calculate_features(data, symbols)
models, predictions = train_return_models(features, symbols)
volatility_data = prepare_volatility_data(data.pct_change().dropna(), symbols)
weights, performance = optimize_portfolio(predictions, volatility_data)
backtest_with_predictions(data, symbols, weights)
```

### Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. For major changes, please open an issue to discuss your proposal.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
