import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten
# Load the dataset
data = pd.read_parquet(r'posedetection_mediapipe\cvzone\test-00000-of-00001.parquet')

# Preprocess the data
def preprocess_data(data):
    X = []
    y = []
    for index, row in data.iterrows():
        image_data = row['image']
        label = row['label']
        
        # Decode the image data
        image_bytes = image_data['bytes']
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Normalize pixel values and resize the image
        img = cv2.resize(img, (100, 100))  # Adjust dimensions as needed
        img = img / 255.0
        
        X.append(img)
        y.append(label)
        
    X = np.array(X)
    y = to_categorical(np.array(y))
    
    return X, y

X, y = preprocess_data(data)

# Display 5 images after preprocessing
def display_images(X):
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(X[i])
        plt.axis('off')
    plt.show()

display_images(X)
# Load the dataset
data = pd.read_parquet(r'posedetection_mediapipe\cvzone\test-00000-of-00001.parquet')

# Preprocess the data
def preprocess_data(data):
    X = []
    y = []
    for index, row in data.iterrows():
        image_data = row['image']
        label = row['label']
        
        # Decode the image data
        image_bytes = image_data['bytes']
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Normalize pixel values and resize the image
        img = cv2.resize(img, (100, 100))  # Adjust dimensions as needed
        img = img / 255.0
        
        X.append(img)
        y.append(label)
        
    X = np.array(X)
    y = to_categorical(np.array(y))
    
    return X, y

X, y = preprocess_data(data)

# Display 5 images after preprocessing
def display_images(X):
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(X[i])
        plt.axis('off')
    plt.show()

display_images(X)

# Define the model
model = Sequential()

# CNN for feature extraction
input_shape = (None, X.shape[2], X.shape[3], X.shape[4])  # Assuming X is of shape (samples, 100, 100, 3)
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=input_shape))
model.add(TimeDistributed(MaxPooling2D((2, 2), padding='same')))
model.add(TimeDistributed(Flatten()))

# LSTM for sequence learning
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))

# Fully connected layer
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=10, batch_size=32)

# Save the model
model.save('exercise_recognition_model.h5')
