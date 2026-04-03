import h5py
import pandas as pd

people = ["thiago", "blake", "ethan"]
activities = ["walking", "jumping"]

with h5py.File("data.h5", "w") as f:
    raw = f.create_group('Raw Data')
    pre = f.create_group('Pre-processed Data')
    seg = f.create_group('Segmented Data')

    train_group = seg.create_group("Train")
    test_group = seg.create_group("Test")

    for p in people: #Iterate through team members
        p_group = raw.create_group(p)

        for a in activities: #Iteratie through activities
            filepath = f"raw_data/{p}_{a}.csv"

            df = pd.read_csv(filepath)
            data = df.to_numpy() #NumPy array

            p_group.create_dataset(a, data=data) #Add to HDF5 file