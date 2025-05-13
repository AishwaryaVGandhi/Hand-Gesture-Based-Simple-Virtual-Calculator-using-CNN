import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from collections import deque
import time


# Load trained digit recognition model
model = tf.keras.models.load_model("digit_model.h5")

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)  # Detect only one hand

# Webcam setup
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.set(3, 1280)
# cap.set(4, 720)

blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
points = deque(maxlen=512)

# Variables for expression handling
expression = ""
writing_mode = False  # True when drawing
last_operator = None  # Prevent duplicate operators
last_operator_time = time.time()  # Cooldown for operators
confirm_start_time = None  # For detecting fist hold time
clear_start_time = None  # For detecting open hand hold time


def preprocess_digit(img):
    # Convert to grayscale and apply adaptive thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Noise removal
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours with area filtering
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 500:
        return None

    # Extract and center the digit
    x, y, w, h = cv2.boundingRect(max_contour)
    digit = thresh[y:y + h, x:x + w]

    # Create square canvas maintaining aspect ratio
    size = max(w, h) + 20  # Add padding
    canvas = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = digit

    # Resize and normalize
    resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    return resized.reshape(1, 28, 28, 1).astype('float32') / 255.0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hands, img = detector.findHands(frame, draw=False)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        for hand in hands:
            lmList = hand["lmList"]  # Landmark points
            connections = detector.mpHands.HAND_CONNECTIONS  # Hand connections

            # Draw hand connections
            for conn in connections:
                pt1, pt2 = lmList[conn[0]][:2], lmList[conn[1]][:2]
                cv2.line(img, pt1, pt2, (255, 255, 255), 2)

            # Draw landmarks
            for lm in lmList:
                cv2.circle(img, tuple(lm[:2]), 5, (0, 0, 255), -1)  # Red landmarks

        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        fingers1 = detector.fingersUp(hand1)
        connections = detector.mpHands.HAND_CONNECTIONS

        # Writing Mode (Only index finger up)
        if fingers == [0, 1, 0, 0, 0]:
            index_finger = hand["lmList"][8][:2]
            points.appendleft((int(index_finger[0]), int(index_finger[1])))
            writing_mode = True

        # Process & Recognize the Digit (Thumb Up)
        elif fingers == [1, 0, 0, 0, 1] and writing_mode:
            if len(points) > 50:  # Ensure enough strokes
                processed = preprocess_digit(blackboard)
                if processed is not None:
                    pred = model.predict(processed)
                    digit = np.argmax(pred)
                    confidence = np.max(pred)

                    if confidence > 0.7:
                        expression += str(digit)
                        writing_mode = False  # Reset writing mode after recognition

                # Reset drawing area
                blackboard.fill(0)
                points.clear()

        # Operator Selection (Cooldown applied)
        current_time = time.time()
        if current_time - last_operator_time > 0.5:  # 0.5s cooldown
            if fingers == [0, 1, 1, 0, 0] and last_operator != "+":
                expression += "+"
                last_operator = "+"
                last_operator_time = current_time
                last_digit = None  # Reset digit so a second number can be taken
            elif fingers == [0, 1, 1, 1, 0] and last_operator != "-":
                expression += "-"
                last_operator = "-"
                last_operator_time = current_time
                last_digit = None
            elif fingers == [0, 1, 1, 1, 1] and last_operator != "*":
                expression += "*"
                last_operator = "*"
                last_operator_time = current_time
                last_digit = None
            elif fingers == [0, 0, 0, 1, 1] and last_operator != "/":
                expression += "/"
                last_operator = "/"
                last_operator_time = current_time
                last_digit = None

        # Confirm & Evaluate Expression (Fist for 1 sec)
        if fingers == [0, 0, 0, 0, 0]:  # Fist detected
            if confirm_start_time is None:
                confirm_start_time = time.time()  # Start timing
            elif time.time() - confirm_start_time > 1:  # If held for 1 second
                if any(op in expression for op in "+-*/"):
                    try:
                        expression = str(eval(expression))  # Evaluate expression
                    except(ZeroDivisionError, SyntaxError):
                        expression = "Error"
                # else:
                #     expression = "Incomplete"
                confirm_start_time = None  # Reset confirm timing

        else:
            confirm_start_time = None  # Reset if fist not held continuously

        # Clear Expression (Open hand for 1 sec)
        if fingers == [1, 1, 1, 1, 1]:
            if clear_start_time is None:
                clear_start_time = time.time()
            elif time.time() - clear_start_time > 1:
                expression = ""
                blackboard.fill(0)
                points.clear()
                last_digit = None
                last_operator = None
                clear_start_time = None  # Reset clear timing

        else:
            clear_start_time = None  # Reset if open hand not held

    # Draw on blackboard
    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)

    # Display Expression & Result
    display_text = f"Expression: {expression}"
    # print(display_text)
    if "Error" not in display_text:
        cv2.putText(frame, display_text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        # print(display_text)
    else:
        cv2.putText(frame, display_text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        print(display_text)

    # Merge frames
    combined = cv2.addWeighted(frame, 0.7, blackboard, 0.3, 0)
    cv2.imshow("Hand Gesture Based Calculator", combined)

    # Exit condition (Escape key)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

