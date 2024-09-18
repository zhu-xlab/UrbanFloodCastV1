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

To get this data, you can process the data using the script provided in `data/process.ipynb`.

## Model Training
To train the flood model, refer to the Jupyter notebook `mainfile.ipynb`. 

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

1. **Process the runoff / ground truth input data by running the `process.ipynb` notebook. This can help downsample the images.**
   
2. **Train the model using the `mainfile.ipynb` notebook.**
