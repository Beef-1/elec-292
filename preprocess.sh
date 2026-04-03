#!/bin/bash

if [ -f "output/preprocess.pdf" ]; then
    echo "Cleaning files..."
    rm preprocessing.pdf
fi

echo "Generating preprocessed data..."
if python3 preprocess.py; then
    mv preprocessing.pdf output/preprocessing.pdf
    echo "Success"
else
    echo "Failure with exit code $?"
fi