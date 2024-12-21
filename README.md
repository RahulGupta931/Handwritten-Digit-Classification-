# Handwritten Digit Classification using Deep Learning

This project focuses on building and training a deep learning model to classify handwritten digits from the MNIST dataset. The MNIST dataset is a benchmark dataset in the field of machine learning and computer vision, containing 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Model Architecture](#model-architecture)

## Project Overview
The goal of this project is to:
- Preprocess and visualize the MNIST dataset.
- Train a deep learning model to achieve high accuracy on test data.
- Evaluate the model's performance and visualize predictions.

## Dataset
The MNIST dataset consists of:
- **Training set:** 60,000 images
- **Test set:** 10,000 images

Each image represents a digit from 0 to 9 and is labeled accordingly. The dataset can be accessed via the [TensorFlow/Keras datasets module](https://www.tensorflow.org/datasets).

## Results
The trained model achieved:

Training Accuracy: 99%
Test Accuracy: 98%
Example Predictions

## Technologies Used
Python
TensorFlow/Keras for model building and training
Matplotlib for data visualization
NumPy for numerical computations
Contributing
Contributions are welcome! Please fork this repository and create a pull request with your changes.

## Model Architecture
The model used for this project includes:
1. **Input Layer**: Accepts 28x28 grayscale images.
2. **Flatten Layer**: Converts the 2D input into a 1D vector.
3. **Hidden Layers**: Fully connected dense layers with ReLU activation.
4. **Output Layer**: 10 neurons (one for each digit), with softmax activation.

### Example Model Summary
```plaintext
Layer (type)                  Output Shape              Param #   
=================================================================
flatten (Flatten)             (None, 784)              0         
dense_1 (Dense)               (None, 128)              100480    
dense_2 (Dense)               (None, 64)               8256      
dense_3 (Dense)               (None, 10)               650       
=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0
