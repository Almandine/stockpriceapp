# Importing the required libraries
import streamlit as st

# What is Streamlit?
# Streamlit is an open-source Python library that makes it easy to build custom web apps for data science, machine learning, and general interactive purposes.
# It is popular because it enables rapid prototyping and deployment of web-based dashboards and applications without the need for extensive web development knowledge.

# Title of the Streamlit App
st.title('Simple Streamlit App')

# Adding some text description
st.write("Hello, this is a simple Streamlit application!")

# A number input box for user interaction
user_input = st.number_input('Enter a number:', value=0)

# Displaying the value entered by the user
st.write(f'You entered: {user_input}')

# Adding a button to the interface
if st.button('Click Me!'):
    st.write('You clicked the button!')

# Explanation
# Streamlit apps run on the local server, and they allow you to build interactive components like buttons, sliders, and text inputs
# simply by using Python code. This makes it a go-to tool for data scientists to showcase their models or explore datasets interactively.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import plotly.graph_objects as go
import io

# App Title
st.title("Stock Price Forecasting App")
st.markdown("This app uses machine learning (LSTM) to predict stock prices for the upcoming week based on historical data fetched from Yahoo Finance.")

# Sidebar for Adjustable Parameters
st.sidebar.header("Adjustable Parameters")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
training_epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=100, value=20)
lstm_layers = st.sidebar.slider("LSTM Layers", min_value=1, max_value=5, value=2)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, step=16, value=32)

# Fetching Data
@st.cache_data
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

st.subheader("Historical Stock Data")
data = fetch_data(stock_ticker, start_date, end_date)
if data is not None and not data.empty:
    st.write(data.tail())
else:
    st.error("Failed to load data. Check the stock ticker or date range.")

# Preprocessing
if data is not None and not data.empty:
    data['Date'] = data.index
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Splitting data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Preparing data for LSTM
    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Building LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    for _ in range(lstm_layers - 1):
        model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Training the model
    with st.spinner("Training the model..."):
        model.fit(X_train, y_train, epochs=training_epochs, batch_size=batch_size, verbose=0)

    # Predicting
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transforming predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Forecasting future prices
    last_60_days = scaled_data[-time_step:]
    X_future = last_60_days.reshape(1, -1, 1)
    future_predictions = []
    for _ in range(7):
        pred = model.predict(X_future)[0][0]
        future_predictions.append(pred)
        X_future = np.append(X_future[:, 1:, :], [[pred]], axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Visualizing
    st.subheader("Stock Price Predictions")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Predicted Prices'))
    fig.update_layout(title=f"{stock_ticker} Stock Price Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Downloading Results
    result_df = pd.DataFrame({
        "Date": list(data.index) + list(future_dates),
        "Actual Price": list(data['Close']) + [None] * 7,
        "Predicted Price": [None] * len(data) + list(future_predictions.flatten())
    })
    csv = io.StringIO()
    result_df.to_csv(csv, index=False)
    st.download_button(label="Download Predictions as CSV", data=csv.getvalue(), file_name="predicted_prices.csv", mime="text/csv")
