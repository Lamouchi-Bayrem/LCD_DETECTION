import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tensorflow.keras.preprocessing import image_dataset_from_directory



# Set Constants
DATA_DIR = r"C:\Users\Administrateur\Desktop\LCD_DIGIT\data"
IMG_SIZE = (28, 28)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Load dataset

val_test_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# Extract class names
class_names = val_test_dataset.class_names

# Fix syntax error in test_batches
test_batches = val_test_dataset.skip(len(val_test_dataset) // 2)

# Load model properly
model = keras.models.load_model("lcd_digit_recognition.h5")

# Function to make predictions on an image
def predict(model, img):
    img = tf.image.rgb_to_grayscale(img)  # Convert RGB to grayscale
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# Test with batch images
plt.figure(figsize=(15, 15))
for images, labels in test_batches.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i])
        actual_class = class_names[labels[i].numpy()] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%", fontsize=10)
        plt.axis("off")

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Add space between images
plt.show()

