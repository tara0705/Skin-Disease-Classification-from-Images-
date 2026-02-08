# ================================
# CNN Architecture & Feature Extraction
# Skin Disease Classification
# ================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

# -------------------------------
# Image & Class Configuration
# -------------------------------
IMG_SIZE = 224
NUM_CLASSES = 7   # HAM10000 dataset

# -------------------------------
# CNN Model Architecture
# -------------------------------
model = Sequential([
    # Input Layer
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # -------- Convolution Block 1 --------
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # -------- Convolution Block 2 --------
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # -------- Convolution Block 3 --------
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    # -------- Fully Connected Layers --------
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    # -------- Output Layer --------
    Dense(NUM_CLASSES, activation='softmax')
])

# -------------------------------
# Compile the Model
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Model Summary
# -------------------------------
model.summary()
