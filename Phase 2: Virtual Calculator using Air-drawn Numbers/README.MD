# Phase 2: Virtual Calculator using Air-drawn Numbers

## Overview
An enhanced calculator that combines digit recognition with gesture controls for full expression evaluation.

## Features
- Hybrid input system:
  - Write digits by finger drawing (digit recognition)
  - Gestures for operators:
    - + : (Index + Middle) Fingers Up
    - - : (Index + Middle + Ring) Fingers Up
    - * : (Index + Middle + Ring + Pinky) Fingers Up
    - / : (Pinky + Ring) Fingers Up
- Advanced controls:
  - **Fist for 1s**: Evaluate expression
  - **Open hand for 1s**: Clear everything
- Real-time feedback and error handling

## Requirements
- Python 3.x
- OpenCV
- TensorFlow/Keras
- MediaPipe
- cvzone
- NumPy

## Usage
1. Run the script: `python Air-drawn Numbers.py`
2. Writing mode: Raise only index finger to draw digits
3. Operators: Use specific finger combinations
4. Calculation: Make a fist for 1 second
5. Clear: Show open palm for 1 second
