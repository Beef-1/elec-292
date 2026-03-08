import h5py
import numpy as np
import pandas as pd

runningT = pd.read_csv("Thiago_Running.csv")
jumpingT = pd.read_csv("Thiago_Jumping.csv")
merged_df = pd.concat([runningT, jumpingT], ignore_index=True)
merged_df.to_csv("Thiago_Raw.csv", index=False)

with h5py.File("data.h5", "w") as f:
    raw= f.create_group('Raw Data')
    pre= f.create_group('Pre-processed Data')
    seg= f.create_group('Segmented Data')
    
    raw.create_dataset("Thiago", data= 'Thiago_Raw.csv')
    raw.create_dataset("Blake", data= 'Blake_Raw.csv')
    raw.create_dataset("Ethan", data= 'Ethan_Raw.csv')
    
    pre.create_dataset("Thiago", data= '')
    pre.create_dataset("Blake", data= '')
    pre.create_dataset("Ethan", data= '')
    
    seg.create_dataset("Thiago", data= '')
    seg.create_dataset("Blake", data= '')
    seg.create_dataset("Ethan", data= '')
    
