import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# ---------- SIGN MEANINGS ----------
sign_use = {
    0: "Speed limit 20 km/h",
    1: "Speed limit 30 km/h",
    2: "Speed limit 50 km/h",
    3: "Speed limit 60 km/h",
    4: "Speed limit 70 km/h",
    5: "Speed limit 80 km/h",
    6: "End of speed limit 80 km/h",
    7: "Speed limit 100 km/h",
    8: "Speed limit 120 km/h",
    9: "No overtaking",
    10: "No overtaking for trucks",
    11: "Priority at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no overtaking",
    42: "End of no overtaking for trucks"
}

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model("model/road_sign_cnn.h5")

# ---------- LOAD CLASS INDICES ----------
with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)

# reverse mapping: index → folder name
index_to_class = {v: k for k, v in class_indices.items()}

st.title("🚦 Road Sign Identifier")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = img.reshape(1, 64, 64, 3)

    prediction = model.predict(img)
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    folder_name = index_to_class[class_id]   # e.g. "38"
    sign_number = int(folder_name)           # 38
    meaning = sign_use[sign_number]

    st.success(f"Prediction class: {sign_number}")
    st.warning(f"🚧 Sign Meaning: {meaning}")
    st.info(f"Confidence: {confidence:.2f}%")
