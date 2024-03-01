from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

# ... (The code for data preprocessing and model training remains the same)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    input_price = float(request.form['input_price'])
    input_data = np.array([input_price]).reshape(1, 1)
    scaled_input = scaler.transform(input_data)
    X_input = scaled_input.reshape(1, look_back, 1)
    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]
    return render_template('index.html', prediction=predicted_price)

if __name__ == '__main':
    app.run(debug=True)
