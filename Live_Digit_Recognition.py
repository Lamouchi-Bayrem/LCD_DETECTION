import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("lcd_digit_recognition.h5")

# Define constants
IMG_SIZE = (28, 28)  # Match your training image size
class_names = [str(i) for i in range(10)]  # Digits 0-9

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define ROI (Region of Interest) - Adjust as needed
    x, y, w, h = 100, 100, 200, 200
    roi = gray[y:y+h, x:x+w]

    # Resize to match model input size
    roi_resized = cv2.resize(roi, IMG_SIZE)
    
    # Normalize and reshape
    img_array = roi_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])  # Shape: (1, 28, 28, 1)

    # Predict digit
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    # Display results
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f"Digit: {predicted_class} ({confidence:.2f}%)",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Live Digit Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
