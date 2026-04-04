import numpy as np
import pandas as pd

PREPROCESS_MA_WINDOW = 3


def fill_missing(data):
    df = pd.DataFrame(data)
    df.interpolate(method="linear", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df.to_numpy()


def moving_average(data, window_size):
    smoothed = np.copy(data).astype(float)
    last_signal_col = min(4, data.shape[1] - 1)
    for col in range(1, last_signal_col + 1):
        series = pd.Series(data[:, col])
        smoothed[:, col] = series.rolling(
            window=window_size,
            min_periods=1,
        ).mean().to_numpy()
    return smoothed
