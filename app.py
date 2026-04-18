import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import onnxruntime as ort

SEQ_LEN = 30
HORIZON = 30

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
@st.cache_resource
def load_all():
    session_gru = ort.InferenceSession(f"gru_model_{SEQ_LEN}.onnx")
    session_lstm = ort.InferenceSession(f"lstm_model_{SEQ_LEN}.onnx")
    scaler = joblib.load(f"scaler_{SEQ_LEN}.pkl")
    df = pd.read_csv(f"historical_data_{SEQ_LEN}.csv", index_col=0, parse_dates=True)
    return session_gru, session_lstm, scaler, df


session_gru, session_lstm, scaler, df_filled = load_all()

FEATURES = [
    "pH",
    "specific_conductance",
    "water_temperature",
    "dissolved_oxygen",
    "discharge",
]

# -----------------------------
# UI
# -----------------------------
st.title("Water Quality Forecast (30 Days)")

st.markdown(
    """
Select a model:

- **GRU + Attention Model (Baseline)**
- **Enhanced LSTM with Attention Mechanism**
"""
)

model_choice = st.selectbox(
    "Choose Model",
    ["GRU + Attention Model (Baseline)", "Enhanced LSTM with Attention Mechanism (Premium)"],
)

if "Enhanced LSTM with Attention Mechanism" in model_choice:
    st.success("Premium model selected: higher accuracy across most variables.")
else:
    st.info("Baseline model selected: simpler and stable predictions.")

feature = st.selectbox("Select parameter", FEATURES + ["WQI"])

def compute_wqi(df):
    # reference values
    T_ref = 25
    T_range = 25
    DO_sat = 8
    pH_ideal = 7

    cond_max = df["specific_conductance"].max()
    cond_max = cond_max if cond_max > 0 else 1

    # indices (0 to 1)
    I_temp = 1 - (abs(df["water_temperature"] - T_ref) / T_range)
    I_do = df["dissolved_oxygen"] / DO_sat
    I_cond = 1 - (df["specific_conductance"] / cond_max)
    I_pH = 1 - (abs(df["pH"] - pH_ideal) / pH_ideal)

    # clip values
    I_temp = I_temp.clip(0, 1)
    I_do = I_do.clip(0, 1)
    I_cond = I_cond.clip(0, 1)
    I_pH = I_pH.clip(0, 1)

    # final WQI (0 to 100)
    wqi = ((I_temp + I_do + I_cond + I_pH) / 4) * 100

    return wqi


def classify_wqi(wqi):
    if wqi >= 90:
        return "Excellent"
    elif wqi >= 70:
        return "Good"
    elif wqi >= 50:
        return "Moderate"
    elif wqi >= 25:
        return "Poor"
    else:
        return "Very Poor"


# -----------------------------
# FORECAST FUNCTION
# -----------------------------
def forecast_30_days(df, session, scaler):
    last_seq = df[FEATURES].iloc[-SEQ_LEN:]

    last_scaled = scaler.transform(last_seq.values)

    X_input = last_scaled.reshape(
        1, SEQ_LEN, len(FEATURES)
    ).astype(np.float32)

    outputs = session.run(None, {"input": X_input})
    future_scaled = outputs[0]

    future_2d = future_scaled.reshape(-1, len(FEATURES))
    future_actual = scaler.inverse_transform(future_2d)

    future_dates = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        periods=HORIZON,
    )

    return pd.DataFrame(
        future_actual,
        index=future_dates,
        columns=FEATURES,
    )


# -----------------------------
# RUN
# -----------------------------
if st.button("Generate 30-Day Forecast"):

    if "GRU" in model_choice:
        session = session_gru
        model_label = "GRU + Attention (Baseline)"
    else:
        session = session_lstm
        model_label = "Enhanced LSTM with Attention (Premium)"

    with st.spinner("Generating forecast..."):
        forecast_df = forecast_30_days(df_filled, session, scaler)

    # -----------------------------
    # COMPUTE WQI
    # -----------------------------
    forecast_df["WQI"] = compute_wqi(forecast_df)

    hist = df_filled.copy()
    hist["WQI"] = compute_wqi(hist)

    # -----------------------------
    # TABLE
    # -----------------------------
    st.subheader(f"Forecast Table — {model_label}")
    st.dataframe(forecast_df)

    # -----------------------------
    # GRAPH
    # -----------------------------
    st.subheader(f"{feature} (History vs Forecast)")

    if feature == "WQI":
        fig = go.Figure()

        # HISTORY
        hist_wqi = hist["WQI"][-60:]

        fig.add_trace(
            go.Scatter(
                x=hist_wqi.index,
                y=hist_wqi,
                mode="lines",
                name="History",
                hovertemplate=(
                    "Date: %{x}<br>"
                    + "WQI: %{y:.2f}<br>"
                    + "Status: %{customdata}"
                ),
                customdata=[classify_wqi(v) for v in hist_wqi],
            )
        )

        # FORECAST
        forecast_wqi = forecast_df["WQI"]

        fig.add_trace(
            go.Scatter(
                x=forecast_wqi.index,
                y=forecast_wqi,
                mode="lines",
                name="Forecast",
                line=dict(dash="dash"),
                hovertemplate=(
                    "Date: %{x}<br>"
                    + "WQI: %{y:.2f}<br>"
                    + "Status: %{customdata}"
                ),
                customdata=[classify_wqi(v) for v in forecast_wqi],
            )
        )

        fig.update_layout(
            title="WQI (History vs Forecast)",
            xaxis_title="Date",
            yaxis_title="WQI",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        chart_df = pd.concat(
            [
                hist[[feature]].rename(columns={feature: "History"}),
                forecast_df[[feature]].rename(columns={feature: "Forecast"}),
            ]
        )
        st.line_chart(chart_df)

    # -----------------------------
    # WQI SUMMARY
    # -----------------------------
    st.subheader("Water Quality Index Summary")

    latest_wqi = forecast_df["WQI"].iloc[-1]

    st.write(f"Latest WQI: {latest_wqi:.2f}")
    st.write("Water Quality Status:", classify_wqi(latest_wqi))
