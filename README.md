
# Car Damage Detection

This project aims to detect types of damages on cars, such as cracks, dents, glass shatters, broken lamps, scratches, and flat tires, using deep learning models. We have implemented **DenseNet**, **Mask R-CNN**, and **Faster R-CNN** models to classify and localize the damage in images.

## Project Overview

Car damage detection can help streamline the claims process for insurance companies and repair centers by quickly assessing the type and extent of vehicle damage. This project utilizes three deep learning models to detect different types of car damages through image analysis.

## Dataset

The dataset includes images of cars with various types of damages. Each image is labeled according to the damage type (e.g., crack, dent, glass shatter, lamp broken, scratch, tire flat). The dataset is split into training, validation, and test sets to evaluate model performance.

## Data Preparation

Each model processes the dataset with specific preprocessing steps:

- **DenseNet**: Preprocessed to meet input size requirements of the DenseNet model.
- **Mask R-CNN** and **Faster R-CNN**: Labeled bounding boxes are used to train the models for both classification and localization of damages.

## Models

### 1. DenseNet
DenseNet is used here for image classification to detect damage types in a car image. This model was chosen for its efficient feature extraction and parameter-sharing capabilities, which enhance classification accuracy.

### 2. Mask R-CNN
Mask R-CNN extends Faster R-CNN to provide pixel-wise damage localization by generating a segmentation mask for each instance. This model is particularly useful for precisely identifying the area of damage on the car.

### 3. Faster R-CNN
Faster R-CNN is employed for object detection, identifying and localizing damage types by drawing bounding boxes around them. It uses a Region Proposal Network (RPN) to improve speed and accuracy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/re1th123/car-damage-detection.git
   cd car-damage-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the `data` folder.

## Process

### 1. Prepare the Dataset
Place the dataset in the `data` folder. Make sure the images and labels are organized according to the requirements of each model.

### 2. Update Dataset Paths
In each Python code file (e.g., `train_densenet.py`, `train_mask_rcnn.py`, `train_faster_rcnn.py`), update the dataset path variable to point to your dataset location.

### 3. Train Models
Train each model by running the corresponding training script:
   ```bash
   python train_densenet.py
   python train_mask_rcnn.py
   python train_faster_rcnn.py
   ```

### 4. Evaluate Models
After training, run the evaluation scripts to check model accuracy and performance:
   ```bash
   python evaluate_densenet.py
   python evaluate_mask_rcnn.py
   python evaluate_faster_rcnn.py
   ```

### 5. Run Prediction
Once the models are trained, you can make predictions on new images using the prediction scripts:
   ```bash
   python predict_densenet.py --image <path_to_image>
   python predict_mask_rcnn.py --image <path_to_image>
   python predict_faster_rcnn.py --image <path_to_image>
   ```

## Results

Detailed evaluation metrics such as accuracy, precision, recall, and IoU are documented for each model. The `results` folder contains images with bounding boxes and segmentation masks overlayed to illustrate the model outputs.

## Acknowledgments

We would like to thank the contributors and the open-source community for their support in building deep learning models and for the datasets used in this project.

---

Feel free to contribute to this project by submitting issues, pull requests, or suggestions.
