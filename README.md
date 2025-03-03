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
1. Directly use the processed data: The pixelized data required for model training and evaluation can be downloaded from test_data folder in the following link:[Pixelized Data](https://syncandshare.lrz.de/getlink/fiPWiw7f7nsxXXPBWVN7g/checkpoints)

Format of the .pt file: [height, width, time, channels]. Channels: H, U, V, runoff, DEM. Runoff and DEM are the two inputs. H, U, V are ground truth, represents water height and velocity.

2. Use .txt files to get .pt data, you can process the data using the script provided in `process.ipynb`. This can help convert .txt to .tif files. After processing, `save_pt.ipynb` can help convert .tif to .pt files. The txt files, DEM and ground truth can be found in test_data folder in [Raw Data](https://syncandshare.lrz.de/getlink/fiPWiw7f7nsxXXPBWVN7g/checkpoints)

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
   
2. **Please change all the path to your local path in `load_model_v2.ipynb`.**

3. **Test the model using the `load_model_v2.ipynb` notebook.**

# DNO: Deep Neural Operator

The **DNO (Deep Neural Operator)** is a neural network model designed for learning and predicting complex physical systems, particularly those involving high-dimensional data such as 3D fields. It combines spectral convolutions, pointwise operations, and multi-layer perceptrons (MLPs) to efficiently model spatial and temporal dependencies in data.

---

## Model Architecture

The DNO model consists of the following key components:

1. **Initial Fully Connected Layers**:
   - `fc`: A linear layer mapping input features (8 dimensions) to a hidden state (16 dimensions).
   - `fc0`: A linear layer mapping the hidden state (16 dimensions) to the input channels (10 dimensions) for the convolutional blocks.

2. **Operator Blocks (`conv0`, `conv7`, `conv8`)**:
   - Each operator block contains:
     - **Spectral Convolution (`SpectralConv3d_Uno`)**: A spectral convolution layer for capturing global spatial dependencies in 3D data.
     - **MLP (`MLP3d`)**: A multi-layer perceptron with 3D convolutions for pointwise feature transformations.
       - `mlp1`: A 3D convolution layer mapping input channels (10 or 20) to intermediate channels (20).
       - `mlp2`: A 3D convolution layer mapping intermediate channels (20) back to output channels (10).
     - **Pointwise Operation (`pointwise_op_3D`)**: A 3D convolution layer for local feature transformations.
     - **Normalization Layer (`InstanceNorm3d`)**: Instance normalization for stabilizing training.

3. **Final Fully Connected Layers**:
   - `fc1`: A linear layer mapping the final hidden state (20 dimensions) to an intermediate state (40 dimensions).
   - `fc2`: A linear layer mapping the intermediate state (40 dimensions) to the output (3 dimensions).

---

## Model Parameters

- **Complex Parameter Count**: 8,937,637
- **Real Parameter Count**: 4,470,437

The model uses a combination of real and complex-valued parameters, with the total number of trainable parameters being approximately 8.94 million (complex) or 4.47 million (real).

---

# Model Evaluation Metrics

This repository provides an overview of key evaluation metrics used in hydrological and machine learning models for flood/runoff prediction.

## Metrics Explained
**General Lp Loss**:
   - For any positive real number \( p \), the Lp Loss is:
     ```math
     \text{Lp Loss} = \| \hat{y} - y \|_p = \left( \sum_{i=1}^n |\hat{y}_i - y_i|^p \right)^{1/p}
     ```
   - Characteristics: Flexible, can be tuned for specific tasks.

---

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

**Test value: `0.3927` → Low error, good performance.**

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

**Test value: `0.9371` → Excellent performance.**

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

**Test value: `0.9684` → Strong correlation between predictions and observations.**

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

**Test values:**
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

