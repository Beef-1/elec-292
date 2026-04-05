import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft, fftfreq

SAMPLE_SIZE = 2000 #longer sample instead of segmented data (5 seconds) to show trends clearly
LOC = [1000, 6000, 10000] #3 separate samples

people = ["thiago", "blake", "ethan"]
activities = ["walking", "jumping"]

#Plot all directional accelerations
def plot_acceleration(data, loc, title):
    end = min(loc + SAMPLE_SIZE, len(data))
    fig, ax = plt.subplots()

    ax.plot(data[loc:end, 1], label="ax")
    ax.plot(data[loc:end, 2], label="ay")
    ax.plot(data[loc:end, 3], label="az")

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Acceleration")

    return fig

#Plot accleration magnitude 
#This one was more useful because varying phone orientation cause
def plot_acceleration_mag(data, loc, title):
    end = min(loc + SAMPLE_SIZE, len(data))
    fig, ax = plt.subplots()

    ax.plot(data[loc:end, 4], label="|a|")

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Acceleration Magnitude")

    return fig


def plot_magnitude_overlay(walking_data, jumping_data, loc, title):
    end_w = min(loc + SAMPLE_SIZE, len(walking_data))
    end_j = min(loc + SAMPLE_SIZE, len(jumping_data))

    fig, ax = plt.subplots()

    ax.plot(walking_data[loc:end_w, 4], label="walking", alpha=0.7)
    ax.plot(jumping_data[loc:end_j, 4], label="jumping", alpha=0.7)

    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Acceleration Magnitude")
    ax.set_title(f"Walking vs Jumping Magnitude - {title}")
    ax.legend()

    return fig

#This one summarizes the distributions of frequencies, shows how strongly periodic the data is
#Expected plot: walking is smooth, so less periodic motion; jumping exhibits strong periodic motion
def plot_freq(data, loc, title):  #See report for more details (with citations)
    #Extract sample and compute sampling rate from time column
    end = min(loc + SAMPLE_SIZE, len(data))
    sample = data[loc:end]
    magnitude = sample[:, 4]

    #Compute sampling rate from time (s) column
    sampling_rate = 10

    #Compute FFT on the magnitude signal
    N = len(magnitude)
    freqs = fftfreq(N, d=1 / sampling_rate)
    spectrum = np.abs(fft(magnitude))

    #Only plot positive frequencies
    positive = freqs > 0

    fig, ax = plt.subplots()
    ax.plot(freqs[positive], spectrum[positive])

    ax.set_title(f"Frequency Spectrum - {title}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 1)

    return fig

#Plots distribution of acceleration values
#Expected plot: Walking occupies lower acceleration, jumping higher accel.
def plot_magnitude_histogram(walking_data, jumping_data, person): 
    fig, ax = plt.subplots()

    ax.hist(walking_data[:, 4], bins=50, alpha=0.5, label="walking")
    ax.hist(jumping_data[:, 4], bins=50, alpha=0.5, label="jumping")

    ax.set_xlabel("Acceleration Magnitude")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Acceleration Magnitude - {person}")
    ax.legend()

    return fig

#Heatmap for acceleration 
#Expected plot: Walking should have lower values for everything, running should show higher variance in specific directions
def plot_correlation_heatmap(data, loc, title): #See report for more details (with citations)
    end = min(loc + SAMPLE_SIZE, len(data))
    axes_data = data[loc:end, 1:4]

    corr = np.corrcoef(axes_data.T)

    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1, 2], ["ax", "ay", "az"])
    ax.set_yticks([0, 1, 2], ["ax", "ay", "az"])
    ax.set_title(f"Axis Correlation - {title}")

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
                fig = plot_magnitude_overlay(walking_data, jumping_data, loc, f"{p} at {loc} samples")
                pdf.savefig(fig)
                plt.close(fig)

            #Activity-specific plots
            for a in activities:
                data = f["Raw Data"][p][a][:]
                title = f"{p} - {a}"

                for loc in LOC:
                    for plot_fn in single_activity_functions:
                        fig = plot_fn(data, loc, f"{title} at {loc} samples")
                        pdf.savefig(fig)
                        plt.close(fig)