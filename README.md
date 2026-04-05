# Walking vs Jumping Detector
## Overview
This project includes a full pipeline for data processing and model training, as well as a graphical interface for labeling data.

See `requirements.txt` for a list of all dependencies.

## How to Run
### 1. Run the Data Processing + Model Training Pipeline 
To process the data and train the model, run:
```
    python main.py
```

This script handles:
- Data preprocessing
- Feature extraction
- Model training

Make sure this step is completed before using the GUI.

### 2. Launch the GUI
After the pipeline has finished, you can open the labeling application by running:
```
    python label_app.py
```
This will launch the graphical interface for labelling CSV files and real-time PhyPhox data.