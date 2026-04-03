import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SAMPLE_SIZE = 1000
LOC = 6000
WINDOW_SIZE = 3

people = ["thiago", "blake", "ethan"]
activities = ["walking", "jumping"]

def fill_missing(data):
    df = pd.DataFrame(data)
    df.interpolate(method='linear', inplace=True) #Handle missing values with valid data on both sides
    df.ffill(inplace=True) #Handles esge case of missing value at end of signal
    df.bfill(inplace=True)  #Handles esge case of missing value at start of signal
    return df.to_numpy()

def moving_average(data, window_size):
    smoothed = np.copy(data).astype(float)
    for col in [1, 2, 3, 4]:  # ax, ay, az, magnitude
        series = pd.Series(data[:, col])
        smoothed[:, col] = series.rolling(
            window=window_size, 
            min_periods=1
        ).mean().to_numpy()
    return smoothed

def remove_outliers(data, threshold=5):
    cleaned = np.copy(data)
    for col in [1, 2, 3, 4]:  # ax, ay, az, magnitude
        col_data = cleaned[:, col]
        mean = np.mean(col_data)
        std = np.std(col_data)
        # Replace outliers with the column mean
        mask = np.abs(col_data - mean) > threshold * std
        cleaned[mask, col] = mean
    return cleaned

def plot_before_after(raw, processed, title):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    labels = ["ax", "ay", "az", "|a|"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        col = i + 1
        ax.plot(raw[LOC:LOC+SAMPLE_SIZE, col],
                label="raw", alpha=0.7, color="gray")
        ax.plot(processed[LOC:LOC+SAMPLE_SIZE, col],
                label="filtered", alpha=0.9, color="blue")
        ax.set_ylabel(label)
        ax.legend()

    axes[0].set_title(f"Raw vs Filtered - {title}")
    axes[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    return fig

with h5py.File("data.h5", "a") as f:
    with PdfPages("preprocessing.pdf") as pdf:
        for p in people:
            for a in activities:
                #Load raw data
                raw = f["Raw Data"][p][a][:]

                #Pre-process
                cleaned = fill_missing(raw)
                if a == "walking":
                    cleaned = remove_outliers(cleaned)
                smoothed = moving_average(cleaned, window_size=WINDOW_SIZE)

                #Save back to HDF5
                path = f"Preprocessed/{p}/{a}"
                if path in f:
                    del f[path]
                f[path] = smoothed

                #Save comparison plot to PDF
                fig = plot_before_after(raw, smoothed, f"{p} - {a}")
                pdf.savefig(fig)
                plt.close(fig)