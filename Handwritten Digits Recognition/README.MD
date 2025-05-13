# Handwritten Digit Recognition using CNN

## Overview
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits drawn in real-time using hand gestures via a webcam. The model is trained on the MNIST dataset.

## Features
- Real-time digit drawing using hand tracking (index finger as a pointer)
- CNN model with 99% test accuracy
- Gesture controls:
  - **Draw**: Use index finger to write digits
  - **Recognize**: Press 'r' to predict the drawn digit
  - **Clear**: Press 'c' to reset the drawing board
  - **Quit**: Press 'q' to exit

## Requirements
- Python 3.x
- OpenCV
- TensorFlow/Keras
- MediaPipe
- NumPy

## Usage
1. Run the script: `python digit_recognition.py`
2. Point your index finger at the camera to draw digits
3. Press 'r' to recognize your drawn digit
4. Press 'c' to clear the board

## Model Architecture
- 2 Conv2D layers (32 and 64 filters)
- 2 MaxPooling layers
- Dropout layers for regularization
- Dense layers (128 and 10 neurons)
- Trained for 10 epochs (99% accuracy)
