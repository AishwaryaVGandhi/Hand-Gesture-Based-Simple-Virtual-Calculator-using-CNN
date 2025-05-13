import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from collections import deque

# Load and preprocess MNIST dataset with proper channel dimension
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Build improved CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save & load model
model.save("digit_model.h5")
model = tf.keras.models.load_model("digit_model.h5")

# Accuracy of model
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0]) #0.02
print('Test accuracy:', score[1]) #0.99


# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
points = deque(maxlen=512)
predicted_digit = None

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand tracking
    results = hands.process(rgb_frame)
    index_finger = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger coordinates
            landmark = hand_landmarks.landmark[8]
            h, w = frame.shape[:2]
            index_finger = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(frame, index_finger, 5, (0, 255, 0), -1)
            points.appendleft(index_finger)

    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)

    # Combine frames
    combined = cv2.addWeighted(frame, 0.7, blackboard, 0.3, 0)

    # Display prediction
    if predicted_digit is not None:
        cv2.putText(combined, f"Prediction: {predicted_digit}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Digit Recognition", combined)

    key = cv2.waitKey(1)
    if key == ord('r'):
        processed = preprocess_digit(blackboard)
        if processed is not None:
            pred = model.predict(processed)
            predicted_digit = np.argmax(pred)
            confidence = np.max(pred)
            print(f"Predicted: {predicted_digit} (Confidence: {confidence:.2f})")
            # Only accept confident predictions
            if confidence < 0.7:
                predicted_digit = "Unsure, Please Write Again."
    elif key == ord('c'):
        blackboard.fill(0)
        points.clear()
        predicted_digit = None
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
