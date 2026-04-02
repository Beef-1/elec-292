#!/bin/bash

if [ -f "visualizations.pdf" ]; then
    echo "Cleaning files..."
    rm visualizations.pdf
fi

echo "Generating PDF file..."
if python3 visualize.py; then
    echo "Success"
else
    echo "Failure with exit code $?"
fi