import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model = joblib.load('D:\\Majidh\\Amrita\\Academics\\Sem 5\\Machine Learning Lab with Python\\P\\Using_XGBoost\\Stock Predictions Model_using_XGBoost.joblib')

st.header('Stock Market Predictor Using XGBoost')

stock =st.text_input('Enter Stock Symnbol', 'BTC-USD')
start = '2018-01-01'
end = '2020-12-31'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

st.subheader('Price vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_200_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

# Split data into training and testing sets
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)
data_test_scale = scaler.fit_transform(data_test)

# Prepare data for XG_Boost
X_train = data_train_scale[:-100]  # Features: all but the last 100 prices
y_train = data_train_scale[100:]   # Targets: the next 100 prices
X_test = data_test_scale[:-100]
y_test = data_test_scale[100:]

# Create an XGBoost model
model = XGBRegressor()  # Experiment with different hyperparameters

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Invert scaling for predictions
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualize predicted vs. actual prices
st.subheader('Predicted vs Original')
fig3 = plt.figure(figsize=(10, 8))
plt.plot(y_pred, 'r', label='Predicted Price')
plt.plot(y_test, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig3)

st.subheader('RMSE(Root Mean Square Error) Encountered by using XgBoost')

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
# Calculate RMSE
rmse = np.sqrt(mse)
# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

# Display results with better styling using Markdown
st.markdown(f"**MSE:** `{mse:.4f}`")
st.markdown(f"**RMSE:** `{rmse:.4f}`")
st.markdown(f"**MAE:** `{mae:.4f}`")