from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import skew

from signal_clean import PREPROCESS_MA_WINDOW, fill_missing, moving_average

LABEL_NAMES = {0: "walking", 1: "jumping"}

SCALER_PATH = "scaler.joblib"
CLASSIFIER_PATH = "classifier.joblib"


def clean_signal(data):
    data = np.asarray(data, dtype=float)
    data = fill_missing(data)
    return moving_average(data, PREPROCESS_MA_WINDOW)


def extract_from_array(window, *, clean_first):
    w = np.asarray(window, dtype=float)
    if clean_first:
        w = clean_signal(w)
    x = np.nan_to_num(w[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(w[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    z = np.nan_to_num(w[:, 3], nan=0.0, posinf=0.0, neginf=0.0)

    abs_acc = np.sqrt(x**2 + y**2 + z**2)

    def features(sig):
        if np.ptp(sig) < 1e-14 or float(np.std(sig)) < 1e-14:
            skew_val = 0.0
        else:
            skew_val = skew(sig)

        return [np.mean(sig), np.std(sig), np.min(sig), np.max(sig), np.max(sig) - np.min(sig), np.median(sig), np.var(sig), skew_val, np.sqrt(np.mean(sig**2)), np.mean(sig**2)]

    return features(x) + features(y) + features(z) + features(abs_acc)


def load_classifier_artifacts(directory):
    directory = Path(directory)
    scaler_file = directory / SCALER_PATH
    clf_file = directory / CLASSIFIER_PATH
    return load(scaler_file), load(clf_file)


def load_csv_as_array(path):
    df = pd.read_csv(path)
    return df.to_numpy(dtype=float)


def predict_windows(data, scaler, classifier, window_size, segment_signal):
    data = np.asarray(data, dtype=float)
    segments = segment_signal(data, window_size)
    if len(segments) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    X = np.array([extract_from_array(w, clean_first=False) for w in segments])
    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    y = classifier.predict(X_scaled).astype(int)
    starts = np.arange(len(y), dtype=int) * window_size
    return starts, y


def merged_label_ranges(data, window_starts, labels, window_size):
    i = 0
    n = len(labels)
    while i < n:
        j = i + 1
        while j < n and int(labels[j]) == int(labels[i]):
            j += 1
        start_row = int(window_starts[i])
        end_row = int(window_starts[j - 1] + window_size)
        end_row = min(end_row, len(data))
        t0 = float(data[start_row, 0])
        t1 = float(data[end_row - 1, 0])
        yield start_row, end_row, t0, t1, LABEL_NAMES[int(labels[i])]
        i = j


def ranges_to_dataframe(rows):
    return pd.DataFrame(rows, columns=["start_row", "end_row_exclusive", "start_time_s", "end_time_s", "activity"])
