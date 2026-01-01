# Video Anomaly Detection using Optical Flow

This project implements a computer vision pipeline to detect anomalies in video sequences. It specifically focuses on identifying irregular movements (like vehicles on pedestrian paths) by learning a normal speed profile from training data and comparing it with real-time test footage.

## 🚀 Overview
The system utilizes two primary computer vision techniques:
1. **Gaussian Background Subtraction (MOG2):** To isolate moving objects from the static background.
2. **Farneback Optical Flow:** To calculate the velocity (magnitude and direction) of every pixel.

An anomaly is flagged if an object's speed exceeds the learned "normal" threshold for that specific area of the frame.

## 🛠 Prerequisites
Ensure you have Python installed, then install the required libraries:
```bash
pip install opencv-python numpy
```

## 📊 Dataset Setup

We use the **UCSD Anomaly Detection Dataset** (specifically `ped1` and `ped2`).

### 1. Download the Dataset

You can download the dataset directly using `wget`:

```bash
# Navigate to your project data folder
mkdir -p data && cd data

# Download UCSD Dataset
wget http://www.svcl.ucsd.edu/datasets/anomaly/UCSD_Anomaly_Dataset.tar.gz

# Extract the file
tar -xvf UCSD_Anomaly_Dataset.tar.gz
```

### 2. Project Structure

Ensure your `DATASETS` path in the notebook matches your local directory:

```text
Anomaly-Detection/
├── data/
│   └── UCSD_Anomaly_Dataset/
│       ├── UCSDped1/
│       └── UCSDped2/
├── Anomaly_Detection_OpticalFlow.ipynb
└── README.md
```

## 💻 How to Run

1. Open the notebook:
```bash
jupyter notebook Anomaly_Detection_OpticalFlow.ipynb
```


1. Set the `DATASETS` variable to your local path.
2. Run the **Training** cells first to build the `max_speed` model.
3. Run the **Inference** cells to see the results.

## ⚙️ Parameters Tuning

* `SENSITIVITY`: Increase this (e.g., 1.5) to be less strict about speed.
* `MIN_AREA`: Increase this to ignore smaller noise/objects.
* `MOTION_DIFF_THRESH`: Adjust this for the sensitivity of the speed deviation.

## 📺 Output Example

The output window shows:

* **Left:** Original grayscale or BGR footage.
* **Right:** Anomaly analysis (Heatmap + Red Bounding Boxes for detected anomalies).
