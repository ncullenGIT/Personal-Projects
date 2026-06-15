"""
AI-Powered Security Analysis Tool
---------------------------------
This script:
1. Pulls historical/current market data for a security
2. Builds a forecast model for future price projections
3. Calculates performance metrics
4. Uses the OpenAI API to generate an investment summary

Required Packages:
pip install yfinance pandas numpy matplotlib scikit-learn statsmodels openai python-dotenv

Environment Variables:
OPENAI_API_KEY=your_openai_api_key
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA 
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# LOAD API KEY
load_dotenv()

client = OpenAI(
    base_url="LINK TO API KEY",
    api_key=os.getenv("API KEY")
)

# USER INPUT
ticker = input("Enter ticker symbol (e.g. AAPL, SPY, TSLA): ").upper()

# DOWNLOAD MARKET DATA
print(f"\nDownloading data for {ticker}...\n")

security = yf.Ticker(ticker)

# Pull 5 years of historical data
df = security.history(period="5y")

if df.empty:
    raise ValueError("No data found for ticker.")

# FEATURE ENGINEERING
df = df[['Close', 'Volume']].copy()

# Moving averages
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# Daily returns
df['Daily_Return'] = df['Close'].pct_change()

# Volatility
df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

df.dropna(inplace=True)

# CURRENT METRICS
current_price = df['Close'].iloc[-1]

annual_return = (
    (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252 / len(df))
) - 1

annual_volatility = df['Daily_Return'].std() * np.sqrt(252)

sharpe_ratio = annual_return / annual_volatility

max_drawdown = (
    (df['Close'] / df['Close'].cummax()) - 1
).min()

# ARIMA FORECAST MODEL
print("Training forecasting model...\n")

prices = df['Close']

model = ARIMA(prices, order=(5,1,0))
model_fit = model.fit()

forecast_days = 30

forecast = model_fit.forecast(steps=forecast_days)

future_dates = pd.date_range(
    start=df.index[-1] + timedelta(days=1),
    periods=forecast_days
)

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': forecast
})

# PLOT RESULTS
plt.figure(figsize=(12,6))

plt.plot(df.index, df['Close'], label='Historical Prices')
plt.plot(
    forecast_df['Date'],
    forecast_df['Forecast'],
    label='30-Day Forecast'
)

plt.title(f"{ticker} Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

plt.show()

# FUNDAMENTAL DATA
info = security.info

market_cap = info.get("marketCap", "N/A")
pe_ratio = info.get("trailingPE", "N/A")
forward_pe = info.get("forwardPE", "N/A")
dividend_yield = info.get("dividendYield", "N/A")
beta = info.get("beta", "N/A")
sector = info.get("sector", "N/A")

# PREPARE AI SUMMARY
forecast_end_price = forecast.iloc[-1]

forecast_return = (
    (forecast_end_price - current_price) / current_price
) * 100

summary_prompt = f"""
You are a professional financial analyst.

Analyze the following security data and provide:
1. A concise investment summary
2. Bullish factors
3. Bearish factors
4. Risk considerations
5. Interpretation of the 30-day forecast
6. Overall outlook

Ticker: {ticker}

Sector: {sector}

Current Price: ${current_price:.2f}

Projected 30-Day Price: ${forecast_end_price:.2f}

Projected Return: {forecast_return:.2f}%

Annualized Return: {annual_return:.2%}

Annualized Volatility: {annual_volatility:.2%}

Sharpe Ratio: {sharpe_ratio:.2f}

Maximum Drawdown: {max_drawdown:.2%}

Market Cap: {market_cap}

Trailing PE Ratio: {pe_ratio}

Forward PE Ratio: {forward_pe}

Dividend Yield: {dividend_yield}

Beta: {beta}
"""

# OPENAI API CALL
print("Generating AI investment summary...\n")

response = client.chat.completions.create(
    model="YOUR MODEL",

    max_completion_tokens=4000,

    messages=[
        {
            "role": "system",
            "content": "You are an expert financial analyst."
        },
        {
            "role": "user",
            "content": summary_prompt
        }
    ],

    temperature=0.5
)

ai_summary = response.choices[0].message.content

# DISPLAY RESULTS
print("=" * 60)
print(f"AI INVESTMENT ANALYSIS FOR {ticker}")
print("=" * 60)

print(f"\nCurrent Price: ${current_price:.2f}")
print(f"Projected 30-Day Price: ${forecast_end_price:.2f}")
print(f"Projected Return: {forecast_return:.2f}%")

print("\nPerformance Metrics")
print("-" * 30)
print(f"Annualized Return: {annual_return:.2%}")
print(f"Annualized Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

print("\nFundamental Metrics")
print("-" * 30)
print(f"Market Cap: {market_cap}")
print(f"Trailing PE: {pe_ratio}")
print(f"Forward PE: {forward_pe}")
print(f"Dividend Yield: {dividend_yield}")
print(f"Beta: {beta}")

print("\nAI SUMMARY")
print("=" * 60)
print(ai_summary)

# ==========================================
# OPTIONAL CSV EXPORT
# ==========================================

forecast_df.to_csv(f"{ticker}_forecast.csv", index=False)

print(f"\nForecast exported to {ticker}_forecast.csv")
