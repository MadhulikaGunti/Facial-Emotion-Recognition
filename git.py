import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Load data
data = pd.read_csv(r'C:\Users\chait\OneDrive\Desktop\facial_emotion_detection\fer2013 dataset\fer2013.csv')

# Extract labels and pixels
labels = data.iloc[:, [0]].values
pixels = data['pixels']
Expressions = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# One-hot encode the labels
labels = to_categorical(labels, len(Expressions))

# Convert pixel strings to grayscale images
images = np.array([np.array(pixel.split(), dtype=int) for pixel in pixels])
images = images / 255.0
images = images.reshape(images.shape[0], 48, 48, 1).astype('float32')

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)

# Function to create the CNN model
def create_convolutional_model(classes):
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Second convolutional layer
    model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))

    # Third convolutional layer
    model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))

    # Fourth convolutional layer
    model.add(Conv2D(256, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))

    # Flatten the output and add fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Output layer
    model.add(Dense(classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Number of classes
classes = 7

# Create and summarize the model
model = create_convolutional_model(classes)
model.summary()
