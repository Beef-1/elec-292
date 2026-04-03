import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft, fftfreq

SAMPLE_SIZE = 500 #longer sample instead of segmented data (5 seconds) to show trends clearly
LOC = [1000, 5500, 10000] #3 separate samples

people = ["thiago", "blake", "ethan"]
activities = ["walking", "jumping"]

#Plot all directional accelerations
def plot_acceleration(data, loc, a, p):
    end = min(loc + SAMPLE_SIZE, len(data))
    segment = data[loc:end]
    time = segment[:, 0] - segment[0, 0]

    fig, ax = plt.subplots()
    ax.plot(time, segment[:, 1], label="ax")
    ax.plot(time, segment[:, 2], label="ay")
    ax.plot(time, segment[:, 3], label="az")

    ax.legend()
    ax.set_title(f"{p} - {a} Acceleration at {loc} samples")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")

    return fig

#Plot accleration magnitude 
#This one was more useful because varying phone orientation cause
def plot_acceleration_mag(data, loc, a, p):
    end = min(loc + SAMPLE_SIZE, len(data))
    segment = data[loc:end]
    time = segment[:, 0] - segment[0, 0]

    fig, ax = plt.subplots()
    ax.plot(time, segment[:, 4], label="|a|")

    ax.legend()
    ax.set_title(f"{p} - {a} Acceleration Magnitude at {loc} samples")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration Magnitude")

    return fig

def plot_magnitude_overlay(walking_data, jumping_data, loc, p):
    end_w = min(loc + SAMPLE_SIZE, len(walking_data))
    end_j = min(loc + SAMPLE_SIZE, len(jumping_data))

    seg_w = walking_data[loc:end_w]
    seg_j = jumping_data[loc:end_j]

    time_w = seg_w[:, 0] - seg_w[0, 0]
    time_j = seg_j[:, 0] - seg_j[0, 0]

    fig, ax = plt.subplots()
    ax.plot(time_w, seg_w[:, 4], label="walking")
    ax.plot(time_j, seg_j[:, 4], label="jumping")

    ax.set_title(f"{p} - Walking and Jumping Acceleration at {loc} samples")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration Magnitude")
    ax.legend()

    return fig

#This one summarizes the distributions of frequencies, shows how strongly periodic the data is
#Expected plot: walking is smooth, so less periodic motion; jumping exhibits strong periodic motion
def plot_freq(data, loc, a, p):
    end = min(loc + SAMPLE_SIZE, len(data))
    sample = data[loc:end]
    magnitude = sample[:, 4]
    sampling_rate = 100

    #Compute FFT on the magnitude signal
    N = len(magnitude)
    freqs = fftfreq(N, d=1 / sampling_rate)
    spectrum = np.abs(fft(magnitude))

    #Only plot positive frequencies
    positive = freqs > 0

    fig, ax = plt.subplots()
    ax.plot(freqs[positive], spectrum[positive])

    ax.set_title(f"{p} - {a} Frequency Spectrum at {loc} samples")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 20)

    return fig

#Plots distribution of acceleration values
#Expected plot: Walking occupies lower acceleration, jumping higher accel.
def plot_magnitude_histogram(walking_data, jumping_data, p):
    fig, ax = plt.subplots()

    ax.hist(walking_data[:, 4], bins=50, alpha=0.5, label="walking")
    ax.hist(jumping_data[:, 4], bins=50, alpha=0.5, label="jumping")

    ax.set_xlabel("Acceleration Magnitude")
    ax.set_ylabel("Count")
    ax.set_title(f"{p} - Distribution of Acceleration Magnitude")
    ax.set_xlim(0, 40)
    ax.legend()

    return fig

#Heatmap for acceleration 
#Expected plot: Walking should have lower values for everything, running should show higher variance in specific directions
def plot_correlation_heatmap(data, loc, a, p):
    end = min(loc + SAMPLE_SIZE, len(data))
    axes_data = data[loc:end, 1:4]

    corr = np.corrcoef(axes_data.T)

    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1, 2], ["ax", "ay", "az"])
    ax.set_yticks([0, 1, 2], ["ax", "ay", "az"])
    ax.set_title(f"{p} - {a} Axis Correlation at {loc} samples")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")

    return fig


single_activity_functions = [plot_acceleration, plot_acceleration_mag, plot_freq, plot_correlation_heatmap]

with h5py.File("data.h5", "r") as f:
    with PdfPages("visualizations.pdf") as pdf:

        for p in people:
            walking_data = f["Raw Data"][p]["walking"][:]
            jumping_data = f["Raw Data"][p]["jumping"][:]

            #Histogram (no loc needed)
            fig = plot_magnitude_histogram(walking_data, jumping_data, p)
            pdf.savefig(fig)
            plt.close(fig)

            #Overlay plots with loc
            for loc in LOC:
                fig = plot_magnitude_overlay(walking_data, jumping_data, loc, p)
                pdf.savefig(fig)
                plt.close(fig)

            #Activity-specific plots
            for a in activities:
                data = f["Raw Data"][p][a][:]

                for loc in LOC:
                    for plot_fn in single_activity_functions:
                        fig = plot_fn(data, loc, a, p)
                        pdf.savefig(fig)
                        plt.close(fig)