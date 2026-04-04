import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from signal_clean import PREPROCESS_MA_WINDOW, fill_missing, moving_average

SAMPLE_SIZE = 1000
LOC = 6000
WINDOW_SIZE = PREPROCESS_MA_WINDOW

people = ["thiago", "blake", "ethan"]
activities = ["walking", "jumping"]

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

if __name__ == "__main__":
    with h5py.File("data.h5", "a") as f:
        with PdfPages("preprocessing.pdf") as pdf:
            for p in people:
                for a in activities:
                    raw = f["Raw Data"][p][a][:]

                    cleaned = fill_missing(raw)
                    if a == "walking":
                        cleaned = remove_outliers(cleaned)
                    smoothed = moving_average(cleaned, window_size=WINDOW_SIZE)

                    path = f"Preprocessed/{p}/{a}"
                    if path in f:
                        del f[path]
                    f[path] = smoothed

                    fig = plot_before_after(raw, smoothed, f"{p} - {a}")
                    pdf.savefig(fig)
                    plt.close(fig)