import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
import streamlit as st
from tensorflow import keras

st.title("Stock Price Prediction with LSTM")
st.sidebar.header("Inputs")
start_date = st.sidebar.text_input("Start Date",'2015-01-01')
end_date = st.sidebar.text_input("End Date",'2025-01-09')
tckr = st.sidebar.text_input("Ticker",'TCS.NS')
st.sidebar.header("Model Settings")
epochs_enter = st.sidebar.number_input("Epochs",0,200,100)


st.markdown(
    """
    For a list of stock tickers, you can visit [Yahoo Finance](https://finance.yahoo.com/lookup/) or other sources.
    """
)

try:
    stock_data = yf.download(tickers = tckr, start=start_date, end=end_date)['Close']
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

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

if st.button("Train Model"):
    st.write("Training LSTM model...")
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(50, input_shape=(look_back, 1)))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs_enter, batch_size=64, verbose=1)
    st.success("Model trained successfully")
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # train_predictions = model.predict(X_train)
    # test_predictions = model.predict(X_test)
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train = scaler.inverse_transform(y_train)
    y_test = scaler.inverse_transform(y_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    # st.write(f"Train RMSE: {train_rmse} \n Test RMSE: {test_rmse}")
    
    
    # print(f'Training RMSE: {train_rmse}')
    # print(f'Testing RMSE: {test_rmse}')
    train_plot = np.empty_like(stock_data)
    train_plot[:,:] = np.nan
    train_plot[look_back:len(train_predictions)+look_back, :] = train_predictions

    test_plot = np.empty_like(stock_data)
    test_plot[:,:] = np.nan
    test_plot[len(train_predictions)+(look_back*2):len(stock_data), :] = test_predictions

    #plt.switch_backend('TkAgg')
    # if st.button("Plot Results"):
    st.write("Plotting Results")
    plt.figure(figsize=(16,8))
    plt.plot(scaler.inverse_transform(stock_data), label='Actual Prices', color='blue')
    plt.plot(train_plot, label='Training Predictions', color='green')
    plt.plot(test_plot, label='Testing Predictions', color='red')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)
    model.save("C:\Laptop remains\STUTI\Programa\Stock Predictor\Saved_model\Stock_model.keras")
    
future_date = st.text_input("Date for prediction","2025-01-10")
if st.button('Predict'):
    future_date_obj = datetime.strptime(future_date, '%Y-%m-%d').date()
    if future_date_obj > datetime.strptime(end_date, '%Y-%m-%d').date():
        # future_steps = (future_date - datetime.strptime(end_date, '%Y-%m-%t').date()).days()
        st.write(f"prediciting for future date: {future_date}")
        last_seq = stock_data[-look_back:]
        model = keras.models.load_model("C:\Laptop remains\STUTI\Programa\Stock Predictor\Saved_model\Stock_model.keras")
        future_price = model.predict(last_seq.reshape(1,look_back,1))
        future_price = scaler.inverse_transform(future_price)
        st.success(f"Predicted Stock Price on {future_date}: {future_price[0][0]:.2f}")
    else:
        st.warning("Future date must be beyond the end date of the dataset.")

