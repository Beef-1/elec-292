import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft, fftfreq

SAMPLE_SIZE = 2000 #15 second sample to show trends clearly
LOC = 6000

people = ["thiago", "blake", "ethan"]
activities = ["walking", "jumping"]

#Plot all directional accelerations
def plot_acceleration(data, title):
    fig, ax = plt.subplots()
    ax.plot(data[LOC:LOC+SAMPLE_SIZE, 1], label="ax")
    ax.plot(data[LOC:LOC+SAMPLE_SIZE, 2], label="ay")
    ax.plot(data[LOC:LOC+SAMPLE_SIZE, 3], label="az")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Acceleration")
    return fig

def plot_magnitude_overlay(walking_data, jumping_data, title):
    fig, ax = plt.subplots()
    ax.plot(walking_data[LOC:LOC+SAMPLE_SIZE, 4], label="walking", color="blue", alpha=0.7)
    ax.plot(jumping_data[LOC:LOC+SAMPLE_SIZE, 4], label="jumping", color="orange", alpha=0.7)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Acceleration Magnitude")
    ax.set_title(f"Walking vs Jumping Magnitude - {title}")
    ax.legend()
    return fig

#Plot accleration magnitude
#This one was more useful because varying phone orientation caused inconsistent acceleration direction
def plot_acceleration_mag(data, title):
    fig, ax = plt.subplots()
    ax.plot(data[LOC:LOC+SAMPLE_SIZE, 4], label="|a|")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Acceleration Magnitude")
    return fig

#This one summarizes the distributions of frequencies, shows how strongly periodic the data is
#Expected plot: walking is smooth, so less periodic motion; jumping exhibits strong periodic motion
def plot_freq(data, title): #I had to search this one up
    #Extract sample and compute sampling rate from time column
    sample = data[LOC:LOC+SAMPLE_SIZE]
    time = sample[:, 0]
    magnitude = sample[:, 4]

    #Compute sampling rate from time (s) column
    sampling_rate = 10

    #Compute FFT on the magnitude signal
    N = len(magnitude)
    freqs = fftfreq(N, d=1/sampling_rate)
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
    ax.hist(walking_data[:, 4], bins=50, alpha=0.5, label="walking", color="blue")
    ax.hist(jumping_data[:, 4], bins=50, alpha=0.5, label="jumping", color="orange")
    ax.set_xlabel("Acceleration Magnitude")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Acceleration Magnitude - {person}")
    ax.legend()
    return fig

#Heatmap for acceleration
#Expected plot: Walking should have lower values for everything, running should show higher variance in specific directions
def plot_correlation_heatmap(data, title):
    axes_data = data[:, 1:4]  # ax, ay, az
    corr = np.corrcoef(axes_data.T)

    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0,1,2], ["ax", "ay", "az"])
    ax.set_yticks([0,1,2], ["ax", "ay", "az"])
    ax.set_title(f"Axis Correlation - {title}")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center")
    return fig

with h5py.File("data.h5", "r") as f:
    with PdfPages("visualizations.pdf") as pdf: #Save everything in a PDF instead of having them pop up on screen
        for p in people:
            walking_data = f["Raw Data"][p]["walking"][:]
            jumping_data = f["Raw Data"][p]["jumping"][:]

            for plot_fn, args in [
                (plot_magnitude_histogram, (walking_data, jumping_data, p)),
                (plot_magnitude_overlay, (walking_data, jumping_data, p)),
            ]:
                fig = plot_fn(*args)
                pdf.savefig(fig)
                plt.close(fig)

            for a in activities:
                data = f["Raw Data"][p][a][:]
                title = f"{p} - {a}"

                for plot_fn, args in [
                    (plot_acceleration, (data, title)),
                    (plot_acceleration_mag, (data, title)),
                    (plot_freq, (data, title)),
                    (plot_correlation_heatmap, (data, title)),
                ]:
                    fig = plot_fn(*args)
                    pdf.savefig(fig)
                    plt.close(fig)