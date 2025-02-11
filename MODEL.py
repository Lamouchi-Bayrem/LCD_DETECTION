import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tensorflow.keras.preprocessing import image_dataset_from_directory


# 1. Set Constants
DATA_DIR = r"C:\Users\Administrateur\Desktop\LCD_DIGIT\data"   
IMG_SIZE = (28, 28)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# 2. Load and preprocess dataset
train_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_test_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Split validation and test set
val_batches = val_test_dataset.take(len(val_test_dataset) // 2)
test_batches = val_test_dataset.skip(len(val_test_dataset) // 2)

# Show class names
class_names = train_dataset.class_names
print("Classes: ", class_names)

# Visualize some dataset images
plt.figure(figsize=(10, 5))
for images, labels in train_dataset.take(1):
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Normalize images
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_batches = val_batches.map(lambda x, y: (normalization_layer(x), y))
test_batches = test_batches.map(lambda x, y: (normalization_layer(x), y))

# Cache, Shuffle and Prefetch
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_batches = val_batches.cache().prefetch(buffer_size=AUTOTUNE)
test_batches = test_batches.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2)
])

# 3. Define the CNN model
def create_model():
    model = keras.Sequential([
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])
    return model

# 4. Compile and train the model
model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=val_batches)

# 5. Evaluate and save the model
loss, acc = model.evaluate(test_batches, verbose=2)
print(f"Test Accuracy: {acc:.4f}")

# Save the model using pickle
with open("lcd_digit_recognition.pkl", "wb") as file:
    pickle.dump(model, file)

# Show accuracy and loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
