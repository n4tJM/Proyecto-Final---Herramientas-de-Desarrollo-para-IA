import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image

# Configuración
DATA_DIR = "path_to_dataset"
IMG_SIZE = (64, 64)

# Cargar y preprocesar datos
def load_data():
    images, labels = [], []
    for label in os.listdir(DATA_DIR):
        for file in os.listdir(os.path.join(DATA_DIR, label)):
            img = Image.open(os.path.join(DATA_DIR, label, file)).resize(IMG_SIZE)
            images.append(np.array(img))
            labels.append(int(label))
    return np.array(images) / 255.0, np.array(labels)

X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(43, activation='softmax')  # Número de clases
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
model.save('traffic_sign_model.h5')
