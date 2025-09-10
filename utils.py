import yfinance as yf
import numpy as np
import pandas as pd
from scipy import signal

def fetch_weekly_nifty(start="2007-09-17", end="2022-01-24", ticker="^NSEI"):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(col).split("_")[0] for col in df.columns]
    dfw = df.resample('W-FRI').agg({'Open':'first','Close':'last'})
    dfw = dfw.dropna().reset_index()
    dfw['t'] = np.arange(len(dfw)) / (len(dfw)-1)
    return dfw[['Date','Open','Close','t']]

def extract_extrema(open_series, order=1):
    arr = np.asarray(open_series)
    max_idx = signal.argrelextrema(arr, np.greater, order=order)[0]
    min_idx = signal.argrelextrema(arr, np.less, order=order)[0]
    idx = np.sort(np.concatenate(([0], max_idx, min_idx, [len(arr)-1])))
    idx = np.unique(idx)
    return idx