#!/bin/bash

if [ -f "data.h5" ]; then
    echo "Cleaning files..."
    rm data.h5
fi

echo "Generating HDF5 file..."
if python3 make_h5.py; then
    echo "Success"
else
    echo "Failure with exit code $?"
fi