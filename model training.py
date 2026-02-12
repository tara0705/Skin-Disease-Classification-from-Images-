import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = "train_metadata.csv"
EPOCHS = 50         
BATCH_SIZE = 32
LEARNING_RATE = 0.001

os.makedirs("training", exist_ok=True)

df = pd.read_csv(DATASET_PATH)
print("Columns:", df.columns.tolist())

df = df.drop(columns=["lesion_id", "image_id"])

LABEL_COL = "localization"
y = df[LABEL_COL]
X = df.drop(LABEL_COL, axis=1)

categorical_cols = X.select_dtypes(include=["object"]).columns
feature_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_encoders[col] = le

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

num_classes = len(np.unique(y))

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "training/model_best.keras",
    monitor="val_accuracy",
    save_best_only=True
)

csv_logger = CSVLogger("training/training_logs.csv")


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, csv_logger]
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

model.save("training/model_final.keras")

metadata = {
    "model_type": "ANN (Tabular Data)",
    "features_used": X.columns.tolist(),
    "label_column": LABEL_COL,
    "num_classes": int(num_classes),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "loss_function": "sparse_categorical_crossentropy",
    "data_split": "70% Train / 15% Val / 15% Test",
    "encoding": "Label Encoding",
    "scaling": "StandardScaler",
    "framework": "TensorFlow / Keras",
    "file_format": ".keras"
}

with open("training/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Training completed successfully")
