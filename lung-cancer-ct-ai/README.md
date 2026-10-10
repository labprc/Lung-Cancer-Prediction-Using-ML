# LungCT-AI: AI-Based Lung Cancer Detection from CT Scan Images Using CNN and Explainable AI

## Subtitle
**Deep Learning Based CT Scan Analysis and Lung Cancer Classification**

## Project Overview
LungCT-AI is a comprehensive, modular, and professional web application designed to automatically classify lung cancer using CT scan images. Built with TensorFlow, Keras, and Streamlit, it dynamically adapts to user-provided datasets, supporting multiple deep learning models (Custom CNN, EfficientNetB0, ResNet50, MobileNetV2), performance comparison, and explainable AI through Grad-CAM.

## Problem Statement
Detecting lung cancer from CT scans requires immense expertise and time. An AI-assisted approach can provide a second opinion or highlight critical regions of interest for radiologists. However, many models operate as "black boxes" and fail to explain their reasoning.

## Objectives
- Build an end-to-end classification pipeline for CT scans.
- Provide multiple CNN models including transfer learning approaches.
- Use Grad-CAM to explain model predictions visually.
- Provide a responsive Streamlit dashboard.
- Adapt dynamically to user-provided datasets without hard-coded classes.

## Features
- **Dataset Adaptive**: Dynamically infers classes and classification type (Binary/Multiclass).
- **CNN Classification & Transfer Learning**: Compares models automatically.
- **Explainable AI (Grad-CAM)**: Visualizes the regions strongly contributing to predictions.
- **Data Augmentation**: Configurable image augmentation.
- **Confidence Analysis**: Probability breakdowns and low-confidence warnings.
- **Dataset Analytics**: Dashboard showing class distribution.

## System Architecture
```
Uploaded CT -> Image Validation -> Preprocessing -> Saved Best Model -> Prediction -> Grad-CAM
```

## Dataset Structure
You must provide your own dataset in `data/train` and `data/test`. Classes will be automatically detected based on subfolder names.
```
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── class_n/
└── test/
    ├── class_1/
    ├── class_2/
    └── class_n/
```

## Dataset Preparation
Place your dataset in the structure shown above. The system supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, and optionally `.dcm` (DICOM).

## Installation
1. Clone this repository (or download it).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
To classify a new CT scan image, simply run:
```bash
python run.py
```
This will open a file dialog allowing you to select an image. The system will then process the image and display a window showing the predicted class, confidence probabilities, and the Grad-CAM heatmap.

## CNN Architecture
The Custom CNN implements Batch Normalization, MaxPooling, and Global Average Pooling with Dropout to reduce overfitting.

## Transfer Learning
EfficientNetB0, ResNet50, and MobileNetV2 are provided as base models with custom dense classification heads.

## Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) automatically finds the final convolutional layer of the chosen model and generates heatmaps over the original CT scan.

## Model Evaluation
Generates confusion matrix, ROC-AUC, Precision, Recall, and F1 scores.

## Limitations
- Performance highly depends on the quality and size of the user-provided dataset.
- Does not segment lesions without specific segmentation models.

## Future Scope
- Implementation of 3D CNNs for volumetric CT data.
- Integration of object detection and segmentation masks.

## Medical Disclaimer
> **Medical Disclaimer**
>
> This application is an educational and research prototype demonstrating deep-learning-based analysis of CT scan images. It is not a medical diagnostic system and must not be used to diagnose, rule out, or make treatment decisions about lung cancer. Model predictions may be incorrect and should not replace evaluation by a qualified healthcare professional.
