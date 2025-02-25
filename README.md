# UrbanFloodCastV2

UrbanFloodCastV2 is the second version repository for Inno_Maus project. This repository contains the necessary scripts for data processing, model training, and the required dependencies to run the project.

## Table of Contents
- [Overview](#overview)
- [Data pixelization](#data-pixelization)
- [Model Training](#model-training)
- [Installation](#installation)
- [Usage](#usage)


## Overview
UrbanFloodCastV2 leverages deep learning techniques to forecast urban flooding based on hydrology data. The project uses a variety of libraries and tools for data preprocessing, model building, and visualization.

## Data pixelization
The pixelized data required for model training and evaluation can be downloaded from test_data folder in the following link:

[Pixelized Data](https://syncandshare.lrz.de/getlink/fiPWiw7f7nsxXXPBWVN7g/checkpoints)

To get this data, you can process the data using the script provided in `process.ipynb`. This can help convert .txt to .tif files. Besides, `save_pt.ipynb` can help convert .tif to .pt files.

## Model Training
We prepared the pre-trained model checkpoints: [Checkpoints](https://syncandshare.lrz.de/getlink/fiPWiw7f7nsxXXPBWVN7g/checkpoints) Please download model.pt and put in the `save` folder.

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zhu-xlab/UrbanFloodCastV1.git
   cd UrbanFloodCastV1

2. **Install the required dependencies: You can install all the required libraries using the `requirements.txt` file:**:
   ```bash
   pip install -r requirements.txt

## Usage
To use the model, execute the following steps:
   
1. **Download test data and pretrained model checkpoints. Put in the corresponding folder.**
   
2. **Please change all the path to your local path in `load_model.ipynb`.**

3. **Test the model using the `load_model.ipynb` notebook.**

# Model Evaluation Metrics

This repository provides an overview of key evaluation metrics used in hydrological and machine learning models for flood/runoff prediction.

## Metrics Explained

### 1. Test (RMSE or MSE)
This value likely represents either the **Root Mean Square Error (RMSE)** or **Mean Squared Error (MSE)**, which measure the deviation between predicted and observed values.

#### **Formula:**
**RMSE:**
```math
RMSE = \sqrt{\frac{1}{n} \sum (y_{pred} - y_{obs})^2}
```
**MSE:**
```math
MSE = \frac{1}{n} \sum (y_{pred} - y_{obs})^2
```
- **Lower RMSE/MSE indicates better model performance.**

**Your value: `0.3927` → Low error, good performance.**

---

### 2. Test_nse (Nash-Sutcliffe Efficiency, NSE)
Measures how well the predicted values fit the observed values.

#### **Formula:**
```math
NSE = 1 - \frac{\sum (y_{obs} - y_{pred})^2}{\sum (y_{obs} - \bar{y}_{obs})^2}
```
- **NSE = 1** → Perfect model performance.
- **NSE > 0.75** → Good model performance.
- **NSE > 0.5** → Acceptable but needs improvement.
- **NSE < 0** → Worse than using the mean.

**Your value: `0.9371` → Excellent performance.**

---

### 3. Test_corr (Correlation Coefficient, R)
Measures the linear relationship between predicted and observed values.

#### **Formula (Pearson Correlation):**
```math
r = \frac{\sum (y_{obs} - \bar{y}_{obs}) (y_{pred} - \bar{y}_{pred})}{\sqrt{\sum (y_{obs} - \bar{y}_{obs})^2} \sqrt{\sum (y_{pred} - \bar{y}_{pred})^2}}
```
- **R = 1** → Perfect positive correlation.
- **R = 0** → No correlation.
- **R = -1** → Perfect negative correlation.

**Your value: `0.9684` → Strong correlation between predictions and observations.**

---

### 4. Test_csi_1, Test_csi_2, Test_csi_3 (Critical Success Index, CSI)
Used for evaluating flood/runoff event detection at different thresholds.

#### **Formula:**
```math
CSI = \frac{TP}{TP + FN + FP}
```
- **TP (True Positive):** Correctly predicted flood/runoff events.
- **FP (False Positive):** Incorrectly predicted flood/runoff events.
- **FN (False Negative):** Missed flood/runoff events.
- **CSI = 1** → Perfect prediction.
- **CSI = 0** → Completely failed prediction.

**Your values:**
- **Test_csi_1 = `0.7291`**
- **Test_csi_2 = `0.7807`**
- **Test_csi_3 = `0.7942`**

**Interpretation: Model is effective in detecting flood events (>0.7 indicates good detection performance).**

---

## Summary

| Metric | Meaning | Your Value | Interpretation |
|--------|---------|------------|----------------|
| RMSE / MSE | Measures prediction error | `0.3927` | Low error, good performance |
| NSE | Model efficiency (1 is best) | `0.9371` | Excellent performance (>0.75) |
| Correlation (R) | Linear relationship strength | `0.9684` | Strong correlation between predicted and observed values |
| CSI (x3) | Accuracy for flood/runoff prediction | `0.7291, 0.7807, 0.7942` | Good event detection (>0.7) |

---

## Conclusion
Your model performs **very well** with:
- **Low prediction error** (Test RMSE/MSE)
- **High accuracy** (NSE, Corr, CSI)
- **Effective flood event prediction** (CSI > 0.7)

### 🚀 Potential Improvements:
1. **Hyperparameter tuning** (e.g., learning rate, model complexity).
2. **Enhancing input data** (higher resolution DEM, improved rainfall data).
3. **Trying advanced models** (e.g., LSTM for time-series runoff predictions).

Feel free to contribute or suggest improvements! 🔥
