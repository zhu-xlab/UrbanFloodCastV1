# UrbanFloodCastV1

UrbanFloodCastV1 is the first version repository for Inno_Maus project. This repository contains the necessary scripts for data processing, model training, and the required dependencies to run the project.

## Table of Contents
- [Overview](#overview)
- [Data pixelization](#data-pixelization)
- [Model Training](#model-training)
- [Installation](#installation)
- [Usage](#usage)


## Overview
UrbanFloodCastV1 leverages deep learning techniques to forecast urban flooding based on hydrology data. The project uses a variety of libraries and tools for data preprocessing, model building, and visualization.

## Data pixelization
The pixelized data required for model training and evaluation can be downloaded from the following link:

[Pixelized Data](https://syncandshare.lrz.de/getlink/fi8DyhtffyaeNgMR8u7dh8/output)

To get this data, you can process the data using the script provided in `data/process.ipynb`. This can help convert .txt to .tif files.

## Model Training
We prepared the pre-trained model checkpoints: [Checkpoints](https://syncandshare.lrz.de/getlink/fiPWiw7f7nsxXXPBWVN7g/checkpoints)

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
To train the model, execute the following steps:
   
1. **Process the runoff / ground truth data by running the `downsample.ipynb` notebook. This can help downsample the images. Downsampled runoff is [here](https://syncandshare.lrz.de/getlink/fiP9XQoFCsWhhva2pwyWmr/runoff)**
   
2. **Put the test ground truth events folder to 'data/val'. Test the model using the `load_model.ipynb` notebook.**

3. **Please change the 'Root_path' to your local path.**

