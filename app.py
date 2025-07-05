import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ticker = 'PEP'

# Download PepsiCo stock data
data = yf.download(ticker, start='2015-01-01', end='2024-12-31')
data.reset_index(inplace=True)

# Flatten multi-level columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [' '.join(col).strip() for col in data.columns.values]

# Show columns for debug
print("DATA COLUMNS:", data.columns)

# Pick actual columns
# Find suffix like ' PEP' in columns
suffix = ''
for col in data.columns:
    if col.startswith('Open') and col != 'Open':
        suffix = col.replace('Open', '')
        break

# Build real feature names with suffix if needed
base_features = ['Open', 'High', 'Low', 'Volume']
features = [f + suffix for f in base_features]
features += ['MA_20', 'MA_50', 'Daily_Return', 'Volatility']

# Target col
close_col = 'Close' + suffix

# Fill missing
data = data.ffill().fillna(0)

# Add features
data['MA_20'] = data[close_col].rolling(20).mean()
data['MA_50'] = data[close_col].rolling(50).mean()
data['Daily_Return'] = data[close_col].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(20).std()
data.dropna(inplace=True)

# Confirm final features
print("USING FEATURES:", features)

X = data[features]
y = data[close_col]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Live data
live_data = yf.download(ticker, period='1d', interval='1m')

if isinstance(live_data.columns, pd.MultiIndex):
    live_data.columns = [' '.join(col).strip() for col in live_data.columns.values]

live_data = live_data.ffill().fillna(0)

live_data['MA_20'] = live_data[close_col].rolling(20).mean()
live_data['MA_50'] = live_data[close_col].rolling(50).mean()
live_data['Daily_Return'] = live_data[close_col].pct_change()
live_data['Volatility'] = live_data['Daily_Return'].rolling(20).std()
live_data.fillna(0, inplace=True)

latest_features = live_data[features].iloc[-1:].dropna()
live_prediction = model.predict(latest_features)

st.title("PepsiCo Stock Price Prediction")

st.line_chart(data[[close_col, 'MA_20', 'MA_50']])
st.write(f"**Predicted Closing Price:** ${live_prediction[0]:.2f}")
