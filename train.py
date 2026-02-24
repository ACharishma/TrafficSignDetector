import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json

# -----------------------------C
# PARAMETERS
# -----------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_DIR = "downlode/Train"
TEST_DIR  = "downlode/Test"

# -----------------------------
# DATA GENERATORS
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_gen.num_classes

# -----------------------------
# MODEL
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -----------------------------
# SAVE MODEL & CLASSES
# -----------------------------
os.makedirs("model", exist_ok=True)


model.save("model/road_sign_cnn.h5")
np.save("model/classes.npy", list(train_gen.class_indices.keys()))

with open("model/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)

print("✅ Model saved")
print("✅ Classes saved")
print("✅ Class indices saved")

