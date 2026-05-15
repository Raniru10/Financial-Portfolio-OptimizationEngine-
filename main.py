import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import warnings

# השתקת אזהרות מערכת כדי לשמור על טרמינל נקי
warnings.filterwarnings('ignore')

print("==================================================")
print("   Markowitz Portfolio Optimization (Max Sharpe)  ")
print("==================================================\n")

# ==========================================
# Phase 1: Data Collection & Processing
# ==========================================
tickers = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'TSLA']
print(f"[*] Downloading historical data for: {tickers}...")

# הורדת הנתונים הגולמיים מיאהו ל-5 השנים האחרונות
raw_data = yf.download(tickers, start="2021-01-01", end="2026-01-01", progress=False)

# שליפת עמודת מחירי הסגירה בלבד
prices = raw_data['Close']

# חישוב תשואות יומיות באחוזים (שבר עשרוני)
returns = prices.pct_change(fill_method=None).dropna()


# ==========================================
# Phase 2: Expected Returns & Covariance Matrix
# ==========================================
print("[*] Calculating Expected Returns and Covariance Matrix...")

# הכפלה ב-252 (ימי מסחר בשנה) כדי לעבור מיומי לשנתי
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252


# ==========================================
# Phase 3: Portfolio Optimization
# ==========================================
print("[*] Initializing Optimization Algorithm (SLSQP)...")

risk_free_rate = 0.04 # ריבית חסרת סיכון (4%)
num_assets = len(tickers)

# פונקציה לחישוב ביצועי התיק (הכריך המתמטי)
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    # חישוב תוחלת התיק (E)
    port_return = np.sum(weights * mean_returns)
    # חישוב סטיית תקן / סיכון התיק (Sigma)
    port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # חישוב יחס שארפ
    sharpe_ratio = (port_return - risk_free_rate) / port_std_dev
    return port_return, port_std_dev, sharpe_ratio

# פונקציית המטרה: מינוס שארפ (כדי שהאלגוריתם יחפש מינימום)
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# הגדרת חוקי המשחק
# חוק 1: סכום המשקלים חייב להיות 1 (100% מהכסף)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# חוק 2: כל משקל מוגבל בין 0.0 ל-1.0
bounds = tuple((0.0, 1.0) for asset in range(num_assets))

# נקודת התחלה אקראית: חלוקה שווה לכל המניות (20% לכל אחת)
initial_guess = num_assets * [1. / num_assets,]

# הרצת האלגוריתם
optimal_result = sco.minimize(negative_sharpe, initial_guess,
                              args=(mean_returns, cov_matrix, risk_free_rate),
                              method='SLSQP', bounds=bounds, constraints=constraints)

# ==========================================
# Results Output
# ==========================================
optimal_weights = optimal_result.x
opt_ret, opt_std, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix, risk_free_rate)

print("\n==================================================")
print("               THE OPTIMAL PORTFOLIO              ")
print("==================================================")
print(f"Expected Annual Return : {opt_ret*100:.2f}%")
print(f"Annual Volatility (Risk) : {opt_std*100:.2f}%")
print(f"Max Sharpe Ratio         : {opt_sharpe:.2f}")

print("\n[*] Optimal Weights Allocation:")
print("--------------------------------")
# הדפסת האחוזים בצורה יפה
for ticker, weight in zip(tickers, optimal_weights):
    print(f" > {ticker}: {weight*100:.2f}%")
print("==================================================\n")