# LCD Digit Recognition

This repository contains a complete pipeline for recognizing LCD-style digits using a Convolutional Recurrent Neural Network (CRNN). The project includes model training, testing, a graphical user interface (GUI), and real-time digit recognition.

## Features
- **Deep Learning Model**: A CRNN trained on LCD digit images for robust recognition.
- **Dataset Handling**: Automatic dataset loading, augmentation, and preprocessing.
- **Testing Pipeline**: Evaluate the trained model on validation and test datasets.
- **Graphical User Interface (GUI)**: An interactive interface for digit prediction.
- **Real-Time Detection**: Live camera feed processing for on-the-fly digit recognition.

---

## Project Structure

```
├── model.py             # Training script for CRNN model
├── test.py              # Script for evaluating the trained model
├── gui.py               # GUI application for digit recognition
├── real_time_detection.py # Real-time LCD digit recognition
├── samples/             # Folder containing sample images for testing
├── README.md            # Project documentation
├── requirements.txt     # List of dependencies
```

---

## Dataset
The dataset used for training the model can be downloaded from Kaggle:
[**LCD Digits Dataset**](https://www.kaggle.com/datasets/markuspfeifer/lcd-digits)

To use the dataset, download it and extract it into a suitable directory before training the model.

---

## Setup Instructions

### **1. Install Dependencies**
Ensure you have Python installed (>=3.8). Install required packages using:
```sh
pip install -r requirements.txt
```

### **2. Train the Model**
Run the following command to train the CRNN model on your dataset:
```sh
python model.py
```
This script loads the dataset, trains the model, and saves it as `crnn_lcd_recognition.h5`.

### **3. Test the Model**
To evaluate the trained model on test data, run:
```sh
python test.py
```
This will generate accuracy metrics and visualize predictions.

### **4. Run the GUI**
To launch the graphical user interface for digit recognition:
```sh
python gui.py
```
The GUI allows users to upload images and receive predictions in real-time.

### **5. Real-Time Detection**
For real-time digit recognition using a webcam, execute:
```sh
python real_time_detection.py
```
This script processes video input and predicts LCD digits in real time.

---

## Model Architecture
The CRNN model combines convolutional layers for feature extraction and recurrent layers for sequence modeling:
1. **Convolutional Layers**: Extract spatial features from digit images.
2. **Max Pooling**: Reduces dimensionality while retaining key features.
3. **Bidirectional LSTM Layers**: Processes sequential patterns in digit structures.
4. **Dense Output Layer**: Predicts digit class probabilities.

---

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- TensorFlow and Keras for deep learning framework.
- OpenCV for real-time image processing.
- The dataset used for training the model from Kaggle.
