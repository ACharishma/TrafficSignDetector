# Traffic Sign Detector 🚦

A deep learning project for traffic sign classification using TensorFlow.

## Overview
This project uses a convolutional neural network (CNN) to classify traffic signs
from images. It is built using TensorFlow and trained on a labeled dataset.

## Tech Stack
- Python
- TensorFlow
- NumPy
- Pandas
- OpenCV

## Project Structure
TrafficSignDetector/
├── app.py
├── train.py
├── preprocess.py
├── sign_mapping.py
├── requirements.txt
└── README.md

## Dataset
The dataset is not included in this repository.

### How to get the dataset:
1. Download the dataset from:
   - (Add dataset link here)

2. Create a folder named:
   downlode/

3. Place the dataset files inside this folder.

## How to Run

### 1. Create virtual environment
python -m venv tfenv
tfenv\Scripts\activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Train the model
python train.py

### 4. Run the application
python app.py

## Notes
- Datasets and virtual environments are excluded from version control.
- Update paths if your dataset location differs.

## Author
Charishma