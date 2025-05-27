import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 48  # Standard size for CNN input

def load_data(dataset_path, label_type):
    images = []
    labels = []
    
    for filename in os.listdir(dataset_path):
        img = cv2.imread(os.path.join(dataset_path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0  # Normalize
        
        if label_type == "age":
            age = int(filename.split("_")[0])  # Extract age from filename
            labels.append(age)
        elif label_type == "emotion":
            emotion = int(filename.split("_")[1])  # Extract emotion label
            labels.append(emotion)
        
        images.append(img)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)

    return images, labels

# Load datasets
X_age, y_age = load_data("dataset/age", "age")
X_emotion, y_emotion = load_data("dataset/emotion", "emotion")

# Convert age to categorical bins
y_age = to_categorical(y_age, num_classes=10)  # 10 Age Groups (0-9, 10-19, ... 90+)

# Convert emotion labels to categorical
y_emotion = to_categorical(y_emotion, num_classes=7)  # 7 Emotion Categories
