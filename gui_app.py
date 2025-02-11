import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
from PIL import Image, ImageTk


# Load the trained model
def load_model():
    with open("lcd_digit_recognition.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
class_names = [str(i) for i in range(10)]  # Assuming 10 classes (0-9)
IMG_SIZE = (28, 28)

# Function to preprocess and predict
def predict_image(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

# GUI Application
def select_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # Load and display image
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img
    
    # Predict and display result
    pred_class, conf = predict_image(file_path)
    result_label.config(text=f"Predicted: {pred_class}\nConfidence: {conf:.2f}%")

# Initialize Tkinter window
root = tk.Tk()
root.title("LCD Digit Recognition")
root.geometry("400x400")

btn = tk.Button(root, text="Select Image", command=select_image)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
