import h5py
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_SIZE = 150 #15 second sample to show trends clearly

with h5py.file("data.h5", "r") as f:
    walking = f["Segmented Data/Train/walking"][:]
    jumping = f["Segmented Data/Train/jumping"][:]

def plot_acceleration(data, title):
    plt.figure()
    plt.plot(data[:, 0], label="ax")
    plt.plot(data[:, 1], label="ay")
    plt.plot(data[:, 2], label="az")
    plt.legend()
    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Acceleration")
    plt.show()
