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

