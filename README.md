# Modern Portfolio Optimization Engine

An algorithmic portfolio optimization tool built in Python. This project utilizes Markowitz's Modern Portfolio Theory (MPT) to find the optimal asset allocation that maximizes the Sharpe Ratio for a given set of stocks.

## Overview
Finding the perfect balance between risk and return is the core of financial mathematics. This engine automates the process by:
1. Fetching historical stock data dynamically via the Yahoo Finance API.
2. Calculating annualized expected returns and the covariance matrix.
3. Using the SLSQP (Sequential Least Squares Programming) optimization algorithm to find the exact portfolio weights that maximize the Sharpe Ratio (global maximum).

## Tech Stack
* Language: Python
* Data Processing: pandas, numpy
* Mathematical Optimization: scipy.optimize
* Financial Data API: yfinance

## Mathematical Background
The engine minimizes the Negative Sharpe Ratio (equivalent to maximizing the positive Sharpe Ratio) using a constrained optimization approach in a convex space:
* Objective Function: Maximize (Expected Return - Risk-Free Rate) / Portfolio Standard Deviation
* Constraints: Sum of all asset weights must equal exactly 1.0 (100%).
* Bounds: No short-selling allowed (each weight 0 <= w <= 1).

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/portfolio-optimization.git](https://github.com/YourUsername/portfolio-optimization.git)
