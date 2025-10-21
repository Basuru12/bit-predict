import sys, threading, io
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.ensemble import GradientBoostingRegressor  # only for type hints
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("Agg")  # render offscreen, then show in Tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -------------------------- Config --------------------------
SYMBOL_DEFAULT = "BTC-USD"
START_DEFAULT  = dt.date(2017, 1, 1)
LAGS           = 1                    # number of lagged 'Close' features
MODEL_PATH     = "assets/gbr_model.joblib"  # <-- put your trained model here
# ------------------------------------------------------------


def resource_path(rel: str) -> Path:
    """
    Works for both dev and PyInstaller-frozen app.
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / rel  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent / rel


def load_model(model_path: str):
    # Shim for older pickles that expect a top-level "_loss" module
    try:
        import sklearn._loss as _skl_loss
        import sys
        sys.modules.setdefault("_loss", _skl_loss)
    except Exception:
        pass

    p = resource_path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    import joblib
    return joblib.load(p)


def fetch_yahoo(symbol: str, start: dt.date, end: dt.date | None = None) -> pd.DataFrame:
    if end is None:
        end = dt.date.today()
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")
    df = df[['Close']].copy()
    df.index = pd.to_datetime(df.index)
    return df


def make_lag_features(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    out = df.copy()
    for k in range(1, lags + 1):
        out[f"Close_lag{k}"] = out["Close"].shift(k)
    # target = current Close, features = prior lags
    out = out.dropna().copy()
    return out


def most_recent_feature_row(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    """
    Build a single-row feature frame for t+1 prediction using the latest lags.
    """
    last_closes = df["Close"].tail(lags).to_numpy()
    if len(last_closes) < lags:
        raise ValueError(f"Need at least {lags} rows to form lag features.")
    # Build feature vector in the same column order used for training
    cols = [f"Close_lag{k}" for k in range(1, lags + 1)]
    # lag1 = most recent, lag2 = previous, ...
    data = {f"Close_lag{k}": last_closes[-k] for k in range(1, lags + 1)}
    return pd.DataFrame([data], columns=cols)


def predict_next_close(model: GradientBoostingRegressor, df: pd.DataFrame, lags: int) -> float:
    X_next = most_recent_feature_row(df, lags)
    y_pred = float(model.predict(X_next)[0])
    return y_pred


def plot_series_with_prediction(df: pd.DataFrame, y_pred: float) -> bytes:
    """
    Returns a PNG bytes of the last 180 days 'Close', plus a marker for the next day prediction.
    """
    lookback = 180
    df_plot = df.tail(lookback).copy()

    # Next business day index (approx)
    next_day = df.index[-1] + pd.tseries.offsets.BDay(1)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(df_plot.index, df_plot["Close"], label="Close (actual)")
    ax.scatter([next_day], [y_pred], marker="o", label=f"Predicted Close ({next_day.date()})")
    ax.set_title("BTC-USD — Close & Next-Day Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.legend(loc="best")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# -------------------------- UI --------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BTC Forecast — GBR (Lagged Close)")
        self.minsize(820, 520)

        # Top controls
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(frm, text="Symbol:").pack(side=tk.LEFT)
        self.symbol_var = tk.StringVar(value=SYMBOL_DEFAULT)
        ttk.Entry(frm, textvariable=self.symbol_var, width=12).pack(side=tk.LEFT, padx=6)

        ttk.Label(frm, text="Start (YYYY-MM-DD):").pack(side=tk.LEFT, padx=(12,0))
        self.start_var = tk.StringVar(value=str(START_DEFAULT))
        ttk.Entry(frm, textvariable=self.start_var, width=12).pack(side=tk.LEFT, padx=6)

        ttk.Button(frm, text="Fetch & Predict", command=self.on_fetch).pack(side=tk.LEFT, padx=10)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var).pack(side=tk.TOP, anchor="w", padx=12)

        # Prediction display
        self.pred_var = tk.StringVar(value="Predicted Close: —")
        pred_lbl = ttk.Label(self, textvariable=self.pred_var, font=("Segoe UI", 12, "bold"))
        pred_lbl.pack(side=tk.TOP, anchor="w", padx=12, pady=(4, 6))

        # Plot canvas placeholder
        self.canvas_holder = ttk.Frame(self)
        self.canvas_holder.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_widget = None  # will hold FigureCanvasTkAgg

        # Pre-load model (once)
        try:
            self.model = load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model error", f"Could not load model:\n{e}")
            self.model = None

    def on_fetch(self):
        if self.model is None:
            messagebox.showwarning("No model", "Model not loaded.")
            return

        symbol = self.symbol_var.get().strip()
        try:
            start = dt.datetime.strptime(self.start_var.get().strip(), "%Y-%m-%d").date()
        except Exception:
            messagebox.showerror("Date error", "Start date must be YYYY-MM-DD.")
            return

        self.status_var.set("Downloading Yahoo data…")
        self.pred_var.set("Predicted Close: —")
        self._set_busy(True)

        threading.Thread(target=self._worker, args=(symbol, start), daemon=True).start()

    def _worker(self, symbol: str, start: dt.date):
        try:
            df = fetch_yahoo(symbol, start)
            # Build lag features for training schema check (optional)
            feat = make_lag_features(df, LAGS)
            # Sanity: ensure model’s expected columns exist
            # (If you trained with exactly these names, it will match.)
            y_pred = predict_next_close(self.model, df, LAGS)
            png = plot_series_with_prediction(df, y_pred)

            # update UI on main thread
            self.after(0, self._update_ui, df.index[-1].date(), y_pred, png)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self.status_var.set("Error. See dialog."))
            self.after(0, lambda: self._set_busy(False))

    def _update_ui(self, last_date, y_pred: float, png_bytes: bytes):
        self.status_var.set(f"Data up to {last_date}.")
        self.pred_var.set(f"Predicted Close: {y_pred:,.2f} USD")
        # draw image via FigureCanvasTkAgg (from raw PNG)
        fig = plt.figure(figsize=(7.5, 4.5), dpi=120)
        ax = fig.add_subplot(111)
        ax.axis("off")
        img = plt.imread(io.BytesIO(png_bytes))
        ax.imshow(img)
        fig.tight_layout()

        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()

        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.canvas_holder)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._set_busy(False)

    def _set_busy(self, busy: bool):
        self.config(cursor="watch" if busy else "")
        for child in self.winfo_children():
            try:
                child.configure(state=("disabled" if busy else "normal"))
            except tk.TclError:
                pass


if __name__ == "__main__":
    App().mainloop()
