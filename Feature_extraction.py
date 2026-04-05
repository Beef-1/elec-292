import h5py
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler

from activity_features import SCALER_PATH, extract_from_array

#Load the segmented data from the HDF5 file for training and testing
with h5py.File("data.h5", "r") as f:  
    X_train_walking = f["Segmented Data/Train/walking"][:]
    X_train_jumping = f["Segmented Data/Train/jumping"][:]
    X_test_walking = f["Segmented Data/Test/walking"][:]
    X_test_jumping = f["Segmented Data/Test/jumping"][:]

#Concatenate the jumping and walking data for training and testing
X_train = np.concatenate([X_train_walking, X_train_jumping], axis=0)
X_test = np.concatenate([X_test_walking, X_test_jumping], axis=0)

#Assign labels as follows: 0 to walking and 1 to jumping
y_train = np.concatenate([
    np.zeros(len(X_train_walking)),
    np.ones(len(X_train_jumping))
])

y_test = np.concatenate([
    np.zeros(len(X_test_walking)),
    np.ones(len(X_test_jumping))
])

#Feature extraction from the X_train and X_test arrays
X_train_feat = np.array([extract_from_array(w, clean_first=True) for w in X_train])
X_test_feat = np.array([extract_from_array(w, clean_first=True) for w in X_test])

#Data normalization using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

#Save the processed data and the scaler for future use
np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

#Save the normalized data for use training the model using joblib
dump(scaler, SCALER_PATH)

#Print the shapes of the normalized data for verification
print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)