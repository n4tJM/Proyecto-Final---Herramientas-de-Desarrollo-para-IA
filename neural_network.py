import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import cv2

# Configuración de parámetros
IMG_SIZE = 32  # Tamaño de las imágenes (32x32 píxeles)
NUM_CLASSES = 10  # Número de clases de señales de tránsito

# Cargar datos del dataset GTSRB
def load_data(data_dir):
    images, labels = [], []
    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, str(class_id))
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_id)
    return np.array(images), np.array(labels)

# Ruta al conjunto de datos
DATASET_DIR = "path_to_gtsrb_dataset"  # Reemplazar con la ruta a tu dataset

# Cargar y dividir los datos
images, labels = load_data(DATASET_DIR)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalización y codificación one-hot
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Definición del modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compilación del modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento del modelo
BATCH_SIZE = 32
EPOCHS = 10

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
    epochs=EPOCHS
)

# Guardar el modelo
model.save('traffic_sign_model.h5')

# Evaluación del modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Accuracy en el conjunto de prueba: {test_acc * 100:.2f}%")

