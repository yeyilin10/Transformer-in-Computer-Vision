# Transformer in Computer Vision

This repository contains the code and documentation for the project on utilizing transformers in computer vision. The project explores the application of Vision Transformer (ViT) and Detection Transformer (DETR) for image classification and object detection tasks respectively.

## Abstract

The transformer model, originally popularized in natural language processing, has shown impressive results in computer vision as well. In this project, we focus on two specific downstream tasks in computer vision and evaluate the performance of Vision Transformer (ViT) and Detection Transformer (DETR) for these tasks. Additionally, we analyze and visualize various features of these transformer models to gain a better understanding of their working principles in computer vision. The results demonstrate that transformers can be effectively leveraged in computer vision applications.

## Features

- Implementation of Vision Transformer (ViT) for image classification
- Implementation of Detection Transformer (DETR) for object detection
- Visualization of attention maps and mean attention distance in ViT
- Evaluation of performance and comparison with traditional CNN-based approaches

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- Other dependencies (specified in requirements.txt)

### Preparation

1. Install the required dependencies:
   pip install -r requirements.txt


### Usage

1. Image Classification with ViT:

- Run the training script for ViT:

  ```
  python train_vit.py
  ```

- Evaluate the trained model:

  ```
  python evaluate_vit.py
  ```

2. Object Detection with DETR:

- Run the training script for DETR:

  ```
  python train_detr.py
  ```

- Evaluate the trained model:

  ```
  python evaluate_detr.py
  ```

### Results

The performance of the models on the respective tasks is as follows:

- ViT Image Classification:

- Precision: 87.549%

- DETR Object Detection:

- Mean Average Precision (mAP): (see report)

For a detailed analysis of the results, please refer to the project report.

## Contributors

- Jianzhe Yu 
- Yufan Chen 
- Yilin Ye 
- Runze Li 

## License

This project is licensed under the [MIT License](LICENSE).
