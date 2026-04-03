#!/bin/bash

if [ -f "output/visualizations.pdf" ]; then
    echo "Cleaning files..."
    rm output/visualizations.pdf
fi

echo "Generating PDF file..."
if python3 visualize.py; then
    mv visualizations.pdf output/visualizations.pdf
    echo "Success"
else
    echo "Failure with exit code $?"
fi