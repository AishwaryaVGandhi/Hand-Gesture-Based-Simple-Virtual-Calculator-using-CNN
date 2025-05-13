# Hand Gesture Based Simple Virtual Calculator using CNN

## Overview
This project combines computer vision and deep learning to create:
1. A handwritten digit recognizer
2. Two versions of gesture-controlled calculators
     - Phase 1: Virtual Calculator using On-screen Buttons
     - Phase 2: Virtual Calculator using Air-drawn Numbers

## System Components
1. **Digit Recognition Module**:
   - CNN model trained on MNIST
   - Real-time drawing and recognition

2. **On-screen Buttons (Phase 1)**:
   - Traditional button-press interface
   - Finger-pointing interaction

3. **Air-drawn Numbers (Phase 2)**:
   - Advanced hybrid input system
   - Combines digit writing and gesture commands
   - No virtual buttons needed

## Technology Stack
- **OpenCV**: Computer vision operations
- **MediaPipe**: Hand tracking and gesture recognition
- **TensorFlow/Keras**: CNN model for digit recognition
- **NumPy**: Numerical operations and array processing
- **cvzone**: Simplified hand tracking interface
- **deque**: Efficient stroke tracking for drawing
- **math**: Distance calculations for gesture detection
- **time**: Cooldown timers and gesture duration tracking

## Installation
```bash 
pip install opencv-python tensorflow mediapipe numpy cvzone
