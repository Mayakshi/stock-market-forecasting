import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Page Config ---
st.set_page_config(page_title="Stock Forecasting App", layout="wide")

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
model_choice = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "Prophet", "LSTM"])
uploaded_file = st.sidebar.file_uploader("Upload Stock Data CSV", type=["csv"])

# --- Theme Styling ---
if theme == "Dark":
    st.markdown("""
        <style>
            .stApp { background-color: #0e1117; color: white; }
            .css-1v0mbdj p, h1, h2, h3 { color: white; }
        </style>
    """, unsafe_allow_html=True)

# --- Load and preprocess ---
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [col.strip().lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    df = df.sort_index()
    return df[['close']]

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_data("Stock_data.csv")  # default file, ensure this exists

# --- Forecasting models ---
def forecast_arima(df):
    model = ARIMA(df['close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    idx = pd.date_range(start=df.index[-1], periods=30, freq='B')
    return pd.Series(forecast, index=idx)

def forecast_prophet(df):
    data = df.reset_index()[['date', 'close']]
    data.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast.set_index('ds')['yhat'].tail(30)

def forecast_lstm(df):
    data = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    input_seq = scaled[-60:].reshape(1, 60, 1)
    forecast = []
    for _ in range(30):
        pred = model.predict(input_seq)[0][0]
        forecast.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    idx = pd.date_range(start=df.index[-1], periods=30, freq='B')
    return pd.Series(forecast, index=idx)

# --- Main App Layout ---
st.title("📈 Stock Market Forecasting Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔮 Forecasting", "📝 Conclusion"])

# Tab 1: Data Overview
with tab1:
    st.subheader("Closing Price History")
    st.line_chart(df['close'])
    st.subheader("Recent Data")
    st.dataframe(df.tail(10), use_container_width=True)

# Tab 2: Forecasting
with tab2:
    st.subheader(f"{model_choice} Forecast")

    with st.expander(f"What is {model_choice}?"):
        if model_choice == "ARIMA":
            st.write("ARIMA combines autoregression, differencing, and moving averages to model time series.")
        elif model_choice == "Prophet":
            st.write("Prophet is designed to handle seasonality and trend changes in time series data.")
        elif model_choice == "LSTM":
            st.write("LSTM is a deep learning model that captures long-term dependencies in sequential data.")

    if model_choice == "ARIMA":
        forecast = forecast_arima(df)
    elif model_choice == "Prophet":
        forecast = forecast_prophet(df)
    else:
        forecast = forecast_lstm(df)

    # Combine last 30 actual points with forecast
    last_30 = df['close'].iloc[-30:]
    combined = pd.concat([last_30, forecast])
    combined.name = "Price"

    max_point = combined.idxmax()
    min_point = combined.idxmin()

    st.subheader("📈 Forecast Chart with Highlights")
    st.line_chart(combined)
    st.markdown(f"🔺 Highest Price: `{combined.max():.2f}` on `{max_point.strftime('%d-%m-%Y')}`")
    st.markdown(f"🔻 Lowest Price: `{combined.min():.2f}` on `{min_point.strftime('%d-%m-%Y')}`")

    # Forecast table with formatted dates & flags
    forecast_df = forecast.to_frame(name="Predicted Close Price")
    forecast_df.index.name = "Date"
    forecast_df.reset_index(inplace=True)
    forecast_df["Date"] = forecast_df["Date"].dt.strftime('%d-%m-%Y')
    forecast_df["Is Highest"] = forecast_df["Predicted Close Price"] == forecast_df["Predicted Close Price"].max()
    forecast_df["Is Lowest"] = forecast_df["Predicted Close Price"] == forecast_df["Predicted Close Price"].min()

    st.subheader("🧾 Forecast Table")
    st.dataframe(forecast_df, use_container_width=True)

# Tab 3: Conclusion
with tab3:
    st.markdown("## ✅ Conclusion")
    st.write("""
This dashboard uses ARIMA, Prophet, and LSTM models to forecast stock prices.

---

### Recommendation:

Start with Prophet for a balance of accuracy and explainability. Use LSTM if your data is large and highly complex.

This tool is perfect for analysts and learners wanting easy, interactive forecasting.

    """)
    st.markdown("📘 **Developed by Mayakshi** · Data Analytics Project · Powered by Streamlit")