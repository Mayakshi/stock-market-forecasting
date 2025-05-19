import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Stock Forecasting App", layout="wide")

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
model_choice = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "Prophet", "LSTM"])
uploaded_file = st.sidebar.file_uploader("Upload Stock Data CSV", type=["csv"])

# --- Apply Theme Styling ---
if theme == "Dark":
    st.markdown("""
        <style>
            .stApp { background-color: #0e1117; color: white; }
            .css-1v0mbdj p, h1, h2, h3 { color: white; }
        </style>
    """, unsafe_allow_html=True)

# --- Load Data ---
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [col.strip().lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    df = df.sort_index()
    return df[['close']]

if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data("Stock_data.csv")

# --- Forecasting Models ---
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

# --- Layout Tabs ---
st.title("📈 Stock Market Forecasting Dashboard")
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔮 Forecasting", "📝 Conclusion"])

# --- Tab 1: Data Overview ---
with tab1:
    st.subheader("Historical Closing Prices")
    st.line_chart(df['close'])
    st.subheader("Recent Data")
    st.dataframe(df.tail(10), use_container_width=True)

# --- Tab 2: Forecasting ---
with tab2:
    st.subheader(f"{model_choice} Forecast")

    with st.expander(f"What is {model_choice}?"):
        st.markdown({
            "ARIMA": "ARIMA models time series using past values and error terms.",
            "Prophet": "Prophet is designed by Meta to capture seasonality and trends.",
            "LSTM": "LSTM is a deep learning model for long-term sequential prediction."
        }[model_choice])

    # Run model
    if model_choice == "ARIMA":
        forecast = forecast_arima(df)
    elif model_choice == "Prophet":
        forecast = forecast_prophet(df)
    else:
        forecast = forecast_lstm(df)

    # Combine last 30 with forecast
    last_30 = df['close'].iloc[-30:]
    combined = pd.concat([last_30, forecast])
    combined.name = "Price"

    max_point = combined.idxmax()
    min_point = combined.idxmin()

    # --- Plotly Chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last_30.index, y=last_30.values, mode='lines', name="Last 30 Days"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name="Forecast"))
    fig.add_trace(go.Scatter(x=[max_point], y=[combined.max()], mode='markers+text', name="Highest",
                             marker=dict(color='green', size=10), text=["High"], textposition="top center"))
    fig.add_trace(go.Scatter(x=[min_point], y=[combined.min()], mode='markers+text', name="Lowest",
                             marker=dict(color='red', size=10), text=["Low"], textposition="bottom center"))
    fig.update_layout(title="📊 Forecast with High/Low Points", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # --- Forecast Table with Highlights ---
    forecast_df = forecast.to_frame(name="Predicted Close Price").copy()
    forecast_df.index.name = "Date"
    forecast_df.reset_index(inplace=True)
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.strftime('%d-%m-%Y')
    forecast_df["Predicted Close Price"] = pd.to_numeric(forecast_df["Predicted Close Price"], errors="coerce")
    forecast_df = forecast_df.dropna(subset=["Predicted Close Price"])
    forecast_df["Is Highest"] = forecast_df["Predicted Close Price"] == forecast_df["Predicted Close Price"].max()
    forecast_df["Is Lowest"] = forecast_df["Predicted Close Price"] == forecast_df["Predicted Close Price"].min()

    def highlight_extremes(row):
        if row["Is Highest"]:
            return ['background-color: #28a745'] * len(row)
        elif row["Is Lowest"]:
            return ['background-color: #dc3545'] * len(row)
        else:
            return [''] * len(row)

    styled_df = forecast_df.style.apply(highlight_extremes, axis=1)

    st.subheader("🧾 Forecast Table (with Highlights)")
    st.dataframe(styled_df, use_container_width=True)

    # --- Download CSV ---
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast as CSV", data=csv, file_name="forecast_output.csv", mime="text/csv")

# --- Tab 3: Conclusion ---
with tab3:
    st.markdown("## ✅ Conclusion")
    st.write("""
This dashboard demonstrates how ARIMA, Prophet, and LSTM can be used to forecast stock prices from historical data.
Start with **Prophet** for balance. Use **LSTM** if data is large or nonlinear.  
Use this tool to quickly experiment with time series forecasting!

""")
    st.markdown("📘 **Developed by Mayakshi** · MSc.BDA Project · Powered by Streamlit")