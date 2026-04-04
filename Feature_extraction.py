import h5py
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler

from activity_features import SCALER_PATH, extract_from_array

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

dump(scaler, SCALER_PATH)

print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)