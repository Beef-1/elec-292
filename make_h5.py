import h5py
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

WINDOW_SIZE = 50

all_segments = {
    "walking": [],
    "jumping": []
}

people = ["thiago", "blake", "ethan"]
activities = ["walking", "jumping"]

def segment_signal(data, window_size):
    segments = []
    for i in range(0, len(data) - window_size, window_size):
        segments.append(data[i:i+window_size])
    return np.array(segments)

with h5py.File("data.h5", "w") as f:
    raw = f.create_group('Raw Data')
    pre = f.create_group('Pre-processed Data')
    seg = f.create_group('Segmented Data')

    train_group = seg.create_group("Train")
    test_group = seg.create_group("Test")

    for p in people:
        p_group = raw.create_group(p)

        for a in activities:
            filepath = f"raw_data/{p}_{a}.csv"

            df = pd.read_csv(filepath)
            data = df.to_numpy()

            p_group.create_dataset(a, data=data)

            segments = segment_signal(data, WINDOW_SIZE)
            all_segments[a].append(segments)
            
    for a in activities:
        combined = np.vstack(all_segments[a])
        combined = shuffle(combined, random_state=42)

        split = int(0.9 * len(combined))
        train = combined[:split]
        test = combined[split:]

        train_group.create_dataset(a, data=train)
        test_group.create_dataset(a, data=test)
    

