import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
start_date = '2015-01-01'
end_date = '2023-10-22'

stock_data = yf.download('msft', start=start_date, end=end_date)['Close']
scaler = MinMaxScaler()
stock_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))
train_size = int(len(stock_data) * 0.8)
train_data, test_data = stock_data[:train_size], stock_data[train_size:]
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 30
X_train, y_train = create_sequences(train_data, look_back)
X_test, y_test = create_sequences(test_data, look_back)
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
train_plot = np.empty_like(stock_data)
train_plot[:,:] = np.nan
train_plot[look_back:len(train_predictions)+look_back, :] = train_predictions

test_plot = np.empty_like(stock_data)
test_plot[:,:] = np.nan
test_plot[len(train_predictions)+(look_back*2):len(stock_data), :] = test_predictions

plt.figure(figsize=(16,8))
plt.plot(scaler.inverse_transform(stock_data), label='Actual Prices', color='blue')
plt.plot(train_plot, label='Training Predictions', color='green')
plt.plot(test_plot, label='Testing Predictions', color='red')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
