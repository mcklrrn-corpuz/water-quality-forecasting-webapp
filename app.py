import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.graph_objects as go
import onnxruntime as ort

SEQ_LEN = 30
HORIZON = 30

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
@st.cache_resource
def load_all():
    session_gru  = ort.InferenceSession(f"gru_model_{SEQ_LEN}.onnx")
    session_lstm = ort.InferenceSession(f"lstm_model_{SEQ_LEN}.onnx")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv(f"historical_data_{SEQ_LEN}.csv", index_col=0, parse_dates=True)
    return session_gru, session_lstm, scaler, df

session_gru, session_lstm, scaler, df_filled = load_all()

FEATURES = [
    'pH',
    'specific_conductance',
    'water_temperature',
    'dissolved_oxygen'
]

# -----------------------------
# UI
# -----------------------------
st.title("Water Quality Forecast (30 Days)")

st.markdown("""
Select a model:

- **LSTM (Baseline)** — stable predictions  
- **GRU + Attention (Premium)** — improved accuracy  
""")

model_choice = st.selectbox(
    "Choose Model",
    ["LSTM (Baseline)", "GRU + Attention (Premium)"]
)

if "GRU" in model_choice:
    st.success("Premium model selected: higher accuracy across most variables.")
else:
    st.info("Baseline model selected: simpler and stable predictions.")

feature = st.selectbox("Select parameter", FEATURES + ["ISQA"])

# ISQA FUNCTION
def compute_isqa(df):
    # constants
    T_ref = 25
    T_range = 25
    DO_sat = 8

    cond_max = df['specific_conductance'].max()
    cond_max = cond_max if cond_max > 0 else 1

    # indices
    I_temp = 1 - (abs(df['water_temperature'] - T_ref) / T_range)
    I_do = df['dissolved_oxygen'] / DO_sat
    I_cond = 1 - (df['specific_conductance'] / cond_max)

    # clip values
    I_temp = I_temp.clip(0, 1)
    I_do = I_do.clip(0, 1)
    I_cond = I_cond.clip(0, 1)

    # modified ISQA
    isqa = I_temp * (I_do + I_cond)

    return isqa


def classify_isqa(isqa):
    if isqa >= 1.5:
        return "Good"
    elif isqa >= 1.0:
        return "Moderate"
    else:
        return "Poor"

# -----------------------------
# FORECAST FUNCTION
# -----------------------------
def forecast_30_days(df, session, scaler):
    last_seq = df[FEATURES].iloc[-SEQ_LEN:]

    # FIX scaler issue
    last_scaled = scaler.transform(last_seq.values)

    X_input = last_scaled.reshape(1, SEQ_LEN, len(FEATURES)).astype(np.float32)

    outputs = session.run(None, {"input": X_input})
    future_scaled = outputs[0]

    future_2d = future_scaled.reshape(-1, len(FEATURES))
    future_actual = scaler.inverse_transform(future_2d)

    future_dates = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        periods=HORIZON
    )

    return pd.DataFrame(
        future_actual,
        index=future_dates,
        columns=FEATURES
    )

# -----------------------------
# RUN
# -----------------------------
if st.button("Generate 30-Day Forecast"):

    if "GRU" in model_choice:
        session = session_gru
        model_label = "GRU + Attention (Premium)"
    else:
        session = session_lstm
        model_label = "LSTM (Baseline)"

    with st.spinner("Generating forecast..."):
        forecast_df = forecast_30_days(df_filled, session, scaler)

    # -----------------------------
    # COMPUTE ISQA
    # -----------------------------
    forecast_df['ISQA'] = compute_isqa(forecast_df)

    hist = df_filled.copy()
    hist['ISQA'] = compute_isqa(hist)

    # -----------------------------
    # TABLE
    # -----------------------------
    st.subheader(f"Forecast Table — {model_label}")
    st.dataframe(forecast_df)

    # -----------------------------
    # GRAPH
    # -----------------------------
    st.subheader(f"{feature} (History vs Forecast)")

    if feature in ["ISQA"]:

        fig = go.Figure()

        # HISTORY
        hist_isqa = hist['ISQA'][-60:]

        fig.add_trace(go.Scatter(
            x=hist_isqa.index,
            y=hist_isqa,
            mode='lines',
            name='History',
            hovertemplate=
            "Date: %{x}<br>" +
            "ISQA: %{y:.2f}<br>" +
            "Status: %{customdata}",
            customdata=[classify_isqa(v) for v in hist_isqa]
        ))

        # FORECAST
        forecast_isqa = forecast_df['ISQA']

        fig.add_trace(go.Scatter(
            x=forecast_isqa.index,
            y=forecast_isqa,
            mode='lines',
            name='Forecast',
            line=dict(dash='dash'),
            hovertemplate=
            "Date: %{x}<br>" +
            "ISQA: %{y:.2f}<br>" +
            "Status: %{customdata}",
            customdata=[classify_isqa(v) for v in forecast_isqa]
        ))

        fig.update_layout(
            title="ISQA (History vs Forecast)",
            xaxis_title="Date",
            yaxis_title="ISQA",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        chart_df = pd.concat([
            hist[[feature]].rename(columns={feature: "History"}),
            forecast_df[[feature]].rename(columns={feature: "Forecast"})
        ])
        st.line_chart(chart_df)

    # ISQA SUMMARY
    st.subheader("Water Quality Index Summary")

    latest_isqa = forecast_df['ISQA'].iloc[-1]

    st.write(f"Latest ISQA: {latest_isqa:.2f}")
    st.write("Water Quality Status:", classify_isqa(latest_isqa))