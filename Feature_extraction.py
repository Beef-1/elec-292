import numpy as np
import h5py
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

def extract_from_array(window):
    x = window[:, 1]
    y = window[:, 2]
    z = window[:, 3]

    abs_acc = np.sqrt(x**2 + y**2 + z**2)

    def feats(sig):
        if np.std(sig) == 0:
            skew_val = 0
        else:
            skew_val = skew(sig)

        return [
            np.mean(sig),
            np.std(sig),
            np.min(sig),
            np.max(sig),
            np.max(sig) - np.min(sig),
            np.median(sig),
            np.var(sig),
            skew_val,
            np.sqrt(np.mean(sig**2)),
            np.mean(sig**2)
        ]

    return feats(x) + feats(y) + feats(z) + feats(abs_acc)

with h5py.File("data.h5", "r") as f:
    X_train_walking = f["Segmented Data/Train/walking"][:]
    X_train_jumping = f["Segmented Data/Train/jumping"][:]
    X_test_walking = f["Segmented Data/Test/walking"][:]
    X_test_jumping = f["Segmented Data/Test/jumping"][:]

X_train = np.concatenate([X_train_walking, X_train_jumping], axis=0)
X_test = np.concatenate([X_test_walking, X_test_jumping], axis=0)

y_train = np.concatenate([
    np.zeros(len(X_train_walking)),
    np.ones(len(X_train_jumping))
])

y_test = np.concatenate([
    np.zeros(len(X_test_walking)),
    np.ones(len(X_test_jumping))
])

X_train_feat = np.array([extract_from_array(w) for w in X_train])
X_test_feat = np.array([extract_from_array(w) for w in X_test])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)