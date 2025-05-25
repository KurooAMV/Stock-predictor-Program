import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import streamlit as st
import keras
import os

# ------------------- UI Setup -------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction with LSTM")

# ------------------- Markdown for ticker -------------------
st.markdown(
    """
    For a list of stock tickers, you can visit [Yahoo Finance](https://finance.yahoo.com/lookup/) or other sources.
    """
)

# ------------------- Sidebar Inputs -------------------
st.sidebar.header("Stock Parameters")
tckr = st.sidebar.text_input("Ticker", "TCS.NS")
start_date = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# st.sidebar.header("Model Settings
epochs_enter = 200
# epochs_enter = st.sidebar.slider("Epochs", 1, 200, 50)
look_back = 30

MODEL_PATH = "saved_model.keras"

# ------------------- Helper Functions -------------------
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)['Close']
    return data

@st.cache_data
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty or 'Close' not in df.columns:
            return None
        df = df[['Close']].dropna()
        df.index = pd.to_datetime(df.index)
        df.columns = ['Closing Price']
        return df
    except Exception as e:
        return None
    
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.LSTM(50, input_shape=input_shape),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# ------------------- Data Loading -------------------

stock_data = fetch_stock_data(tckr, start_date, end_date)

if stock_data is None:
    st.error(f"❌ Could not retrieve data for {tckr}. Try checking the ticker symbol or change the date range.")
else:
    # st.success(f"✅ Data loaded for {tckr}")
    st.line_chart(stock_data)
    # st.dataframe(stock_data.tail())

# ------------------- Data Preprocessing -------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

X_train, y_train = create_sequences(train_data, look_back)
X_test, y_test = create_sequences(test_data, look_back)

# ------------------- Model Training -------------------
if st.button("Train Model"):
    with st.spinner("Training LSTM model..."):
        model = build_model((look_back, 1))
        model.fit(X_train, y_train, epochs=epochs_enter, batch_size=64, verbose=0)
        model.save(MODEL_PATH)
        st.success("Model trained and saved.")

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Inverse transform for plotting
        train_predictions = scaler.inverse_transform(train_predictions)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions))
        st.metric("Train RMSE", f"{train_rmse:.2f}")
        st.metric("Test RMSE", f"{test_rmse:.2f}")

        # Plotting
        total_plot = np.empty_like(scaled_data)
        total_plot[:, :] = np.nan
        total_plot[look_back:len(train_predictions)+look_back] = train_predictions
        total_plot[len(train_predictions)+(look_back*2):] = test_predictions

        plt.figure(figsize=(16, 6))
        plt.plot(scaler.inverse_transform(scaled_data), label='Actual Price', color='blue')
        plt.plot(total_plot, label='Predictions', color='orange')
        plt.legend()
        # st.pyplot(plt)
        st.pyplot(plt.gcf())
        model.save(MODEL_PATH)

# ------------------- Future Prediction -------------------
tomorrow = datetime.today() + timedelta(days=1)
future_date = st.text_input("Future Date (YYYY-MM-DD)", tomorrow.strftime("%Y-%m-%d"))

if st.button("Predict Future Price"):
    try:
        future_date_obj = datetime.strptime(future_date, '%Y-%m-%d').date()
        if future_date_obj <= end_date:
            st.warning("Choose a date beyond the dataset end date.")
        else:
            if not os.path.exists(MODEL_PATH):
                st.error("Model not trained yet. Please train it first.")
                st.stop()

            model = keras.models.load_model(MODEL_PATH)
            last_seq = scaled_data[-look_back:]
            future_price = model.predict(last_seq.reshape(1, look_back, 1))
            future_price = scaler.inverse_transform(future_price)
            st.success(f"Predicted Price on {future_date}: ₹{future_price[0][0]:.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")