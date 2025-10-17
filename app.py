# app.py — Personal Bitcoin Forecast (uses Yahoo Finance)
import io
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Optional deps (for models)
try:
    import joblib
except Exception:
    joblib = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit page setup
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bitcoin Forecast", page_icon="₿", layout="wide")
st.title("₿ Bitcoin Price Forecast (Personal Use)")

st.markdown("""
This app fetches **BTC-USD** data directly from Yahoo Finance, creates time-series
features, and uses your uploaded model (`.pkl`, `.joblib`, or `.onnx`) to forecast
future prices.
""")

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    start_date = st.date_input("Start date", value=date(2017, 1, 1))
    end_date = st.date_input("End date (inclusive)", value=date.today())
    symbol = st.text_input("Ticker", value="BTC-USD")

    st.markdown("### Forecast Settings")
    horizon = st.number_input("Days to forecast", min_value=1, max_value=365, value=14)

    st.markdown("### Model Upload")
    model_file = st.file_uploader("Upload Model (.pkl/.joblib/.onnx)", type=["pkl", "joblib", "onnx"])

    st.markdown("### Feature Settings")
    max_lag = st.slider("Max lag (days)", 1, 30, 7)
    use_rolling = st.checkbox("Add rolling features", value=True)
    win_short = st.number_input("Short MA window", 3, 60, 7)
    win_long = st.number_input("Long MA window", 5, 180, 21)
    win_vol = st.number_input("Volatility window", 3, 60, 7)

    st.markdown("### Feature columns")
    default_feats = ["Prev_Close", "Return_1d", "MA_Short", "MA_Long", "Vol_Short"]
    custom_feats = st.text_input("Columns to use", value=", ".join(default_feats))

# ────────────────────────────────────────────────────────────────────────────────
# Yahoo Finance data loader
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_yahoo_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download price data from Yahoo Finance."""
    df = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),  # yfinance end is exclusive
        auto_adjust=False,
        progress=False
    )
    if df.empty:
        return pd.DataFrame()
    df = df.rename_axis("Date").reset_index()
    return df

df = load_yahoo_data(symbol, start_date, end_date)
if df.empty:
    st.error("No data returned. Try changing the ticker or date range.")
    st.stop()

st.subheader("Raw BTC Data (tail)")
st.dataframe(df.tail(10), use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ────────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   max_lag: int,
                   use_rolling: bool,
                   win_short: int,
                   win_long: int,
                   win_vol: int) -> pd.DataFrame:
    x = df.copy()
    x["Prev_Close"] = x["Close"].shift(1)
    x["Return_1d"] = x["Close"].pct_change()
    if use_rolling:
        x["MA_Short"] = x["Close"].rolling(win_short, min_periods=1).mean()
        x["MA_Long"] = x["Close"].rolling(win_long, min_periods=1).mean()
        x["Vol_Short"] = x["Return_1d"].rolling(win_vol, min_periods=2).std()
    for lag in range(2, max_lag + 1):
        x[f"lag_{lag}"] = x["Close"].shift(lag)
    return x

feat_df = build_features(df, max_lag, use_rolling, win_short, win_long, win_vol)

requested_feats = [c.strip() for c in custom_feats.split(",") if c.strip()]
available_feats = [c for c in requested_feats if c in feat_df.columns]
if not available_feats:
    st.error("No valid feature columns found.")
    st.stop()

model_frame = feat_df[["Date", "Close"] + available_feats].dropna().reset_index(drop=True)
st.subheader("Engineered Features (tail)")
st.dataframe(model_frame.tail(10), use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────────────────
class ModelAdapter:
    def __init__(self, file: io.BytesIO, name: str):
        suffix = Path(name).suffix.lower()
        self.kind = None
        if suffix in [".pkl", ".joblib"]:
            if joblib is None:
                raise RuntimeError("joblib not installed.")
            self.model = joblib.load(file)
            self.kind = "sklearn"
        elif suffix == ".onnx":
            if ort is None:
                raise RuntimeError("onnxruntime not installed.")
            bytes_data = file.read()
            self.ort = ort.InferenceSession(bytes_data, providers=["CPUExecutionProvider"])
            self.onnx_input_name = self.ort.get_inputs()[0].name
            self.kind = "onnx"
        else:
            raise ValueError("Unsupported model format.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "sklearn":
            return np.asarray(self.model.predict(X))
        out = self.ort.run(None, {self.onnx_input_name: X.astype(np.float32)})[0]
        return out.reshape(-1)

def try_get_feature_names(model) -> list[str] | None:
    try:
        return list(getattr(model, "feature_names_in_", []))
    except Exception:
        return None

def align_features(X_df: pd.DataFrame, feature_order: list[str] | None) -> np.ndarray:
    if feature_order:
        return X_df[feature_order].to_numpy(dtype=np.float32)
    return X_df.to_numpy(dtype=np.float32)

def recursive_forecast(last_hist_df, base_df, feature_cols, model, steps,
                       use_rolling, win_short, win_long, win_vol, max_lag):
    preds, dates = [], []
    hist = base_df.copy()
    last_date = pd.to_datetime(hist["Date"].iloc[-1]).date()
    feat_order = None
    if getattr(model, "kind", None) == "sklearn" and getattr(model.model, "feature_names_in_", None) is not None:
        feat_order = list(model.model.feature_names_in_)
    work = last_hist_df[["Date", "Close"]].copy()

    for i in range(steps):
        next_date = last_date + timedelta(days=i + 1)
        tmp = build_features(work, max_lag, use_rolling, win_short, win_long, win_vol)
        tmp2 = tmp[["Date", "Close"] + feature_cols].dropna().reset_index(drop=True)
        X_last = tmp2[feature_cols].tail(1)
        X_mat = align_features(X_last, feat_order)
        y_hat = float(model.predict(X_mat)[-1])
        work = pd.concat([work, pd.DataFrame({"Date": [pd.Timestamp(next_date)], "Close": [y_hat]})], ignore_index=True)
        preds.append(y_hat)
        dates.append(next_date)

    return pd.DataFrame({"Date": dates, "Pred_Close": preds})

# ────────────────────────────────────────────────────────────────────────────────
# Forecasting
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("Forecast")

use_baseline = False
model_adapter = None
model_feat_order = None

if model_file is not None:
    try:
        model_adapter = ModelAdapter(model_file, model_file.name)
        if model_adapter.kind == "sklearn":
            model_feat_order = try_get_feature_names(model_adapter.model)
        st.success(f"Loaded model: {model_file.name}")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        use_baseline = True
else:
    st.info("No model uploaded — using baseline forecast (flat price).")
    use_baseline = True

hist_df = model_frame.copy()
feature_cols = available_feats

try:
    if use_baseline:
        last_close = float(hist_df["Close"].iloc[-1])
        fc = pd.DataFrame({
            "Date": [pd.to_datetime(hist_df["Date"].iloc[-1]).date() + timedelta(days=i+1) for i in range(horizon)],
            "Pred_Close": [last_close] * horizon
        })
    else:
        fc = recursive_forecast(
            last_hist_df=df[["Date", "Close"]],
            base_df=hist_df,
            feature_cols=feature_cols,
            model=model_adapter,
            steps=horizon,
            use_rolling=use_rolling,
            win_short=win_short,
            win_long=win_long,
            win_vol=win_vol,
            max_lag=max_lag,
        )
except Exception as e:
    st.error(f"Forecast failed: {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Plot results
# ────────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(pd.to_datetime(df["Date"]), df["Close"], label="Actual")
ax.plot(pd.to_datetime(fc["Date"]), fc["Pred_Close"], "--", label="Forecast")
ax.set(title=f"{symbol} — Actual vs Forecast", xlabel="Date", ylabel="Price (USD)")
ax.legend()
st.pyplot(fig, clear_figure=True)

# ────────────────────────────────────────────────────────────────────────────────
# Output table + download
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("Forecast table")
st.dataframe(fc, use_container_width=True)

csv = fc.to_csv(index=False).encode()
st.download_button("Download predictions as CSV", data=csv, file_name="btc_forecast.csv", mime="text/csv")

# ────────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────────────────────────
with st.expander("Diagnostics & Info"):
    st.write("Selected features:", feature_cols)
    if model_adapter and getattr(model_adapter, "kind", None) == "sklearn":
        st.write("Model feature_names_in_:", model_feat_order)
    st.write("Rows in engineered frame:", len(model_frame))
