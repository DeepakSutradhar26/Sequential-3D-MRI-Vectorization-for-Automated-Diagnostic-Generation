# 3D CNN-LSTM Hybrid for Volumetric Sequence Classification with Attention-based Feature Enhancement

## Project Overview
This project implements a deep learning framework for classifying 3D volumetric sequences, such as medical imaging data, by combining **3D Convolutional Neural Networks (CNNs)** with **Long Short-Term Memory (LSTM)** networks. The architecture incorporates **Squeeze-and-Excitation (SE) blocks** to enhance important channel-wise features, capturing both spatial and temporal dependencies in the data.  

The model is implemented in **PyTorch** and is designed for end-to-end training and evaluation, including batch processing, GPU acceleration, and performance visualization.

---

## Features
- 3D CNNs for spatial feature extraction from volumetric data  
- LSTM layers for temporal sequence modeling  
- Squeeze-and-Excitation (SE) blocks for attention-based feature enhancement  
- Modular design allowing multiple CNN architectures  
- Training and evaluation pipeline with loss and accuracy tracking  
- Visualization of training and validation loss curves  

---

## Model Architecture

### CNN Architecture
