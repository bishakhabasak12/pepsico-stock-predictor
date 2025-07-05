
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fetch PepsiCo stock data
ticker = 'PEP'
data = yf.download(ticker, start='2015-01-01', end='2024-12-31')
data.reset_index(inplace=True)

# Data Cleaning
data.fillna(method='ffill', inplace=True)
data.fillna(0, inplace=True)

# Feature Engineering
data['MA_20'] = data['Close'].rolling(20).mean()
data['MA_50'] = data['Close'].rolling(50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(20).std()
data.dropna(inplace=True)

# EDA - Plot
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Close')
plt.plot(data['Date'], data['MA_20'], label='MA 20')
plt.plot(data['Date'], data['MA_50'], label='MA 50')
plt.legend()
plt.title('PepsiCo Stock Prices with Moving Averages')
plt.show()

# Correlation heatmap
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# ML Model
features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']

target = 'Close'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")

# Live Prediction
live_data = yf.download(ticker, period='1d', interval='1m')
live_data['MA_20'] = live_data['Close'].rolling(20).mean()
live_data['MA_50'] = live_data['Close'].rolling(50).mean()
live_data['Daily_Return'] = live_data['Close'].pct_change()
live_data['Volatility'] = live_data['Daily_Return'].rolling(20).std()
live_data.fillna(0, inplace=True)

latest_features = live_data[features].iloc[-1:].dropna()
live_prediction = model.predict(latest_features)

print(f"Predicted Closing Price: ${live_prediction[0]:.2f}")
