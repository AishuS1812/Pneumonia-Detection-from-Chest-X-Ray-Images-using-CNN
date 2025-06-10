# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 16:48:44 2025

@author: HP
"""
## Step -- 1
folder_path = r"C:\Users\HP\Desktop\Pneumonia_Project"
import os

folder_path = r"C:\Users\HP\Desktop\Pneumonia_Project"

for file_name in os.listdir(folder_path):
    print(file_name)
##Step -- 2
import os

base_path = r"C:\Users\HP\Desktop\Pneumonia_Project"
##Step -- 3
for split in ['train', 'test', 'val']:
    folder = os.path.join(base_path, split)
    classes = os.listdir(folder)
    print(f"\nContents of '{split}':")
    for c in classes:
        print(f"- {c} ({len(os.listdir(os.path.join(folder, c)))} images)")
##Step -- 4
import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=os.path.join(base_path, 'train'),
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'  # or 'binary' if 2 classes
)
##Step -- 5
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# Set base directory path
base_dir = r"C:\Users\HP\Desktop\Pneumonia_Project"

# Set image size and batch size
img_size = (224, 224)
batch_size = 32

# Load datasets
train_ds = image_dataset_from_directory(
    os.path.join(base_dir, 'train'),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'  # for two classes
)

val_ds = image_dataset_from_directory(
    os.path.join(base_dir, 'val'),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

test_ds = image_dataset_from_directory(
    os.path.join(base_dir, 'test'),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Check class names
print("Class Names:", train_ds.class_names)

# Preview some images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_index = tf.argmax(labels[i]).numpy()
        plt.title(train_ds.class_names[label_index])
        plt.axis("off")
##Step -- 6
from tensorflow.keras import layers

# Image size and batch size
img_size = (224, 224)
batch_size = 32

# 🔄 Data Augmentation (only on training data)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# 📉 Normalize pixel values (0-255 → 0-1)
normalization_layer = layers.Rescaling(1./255)

# 🧠 Apply preprocessing to the datasets
def prepare_dataset(ds, training=False):
    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load and prepare datasets
train_ds_raw = image_dataset_from_directory(
    os.path.join(base_dir, 'train'),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds_raw = image_dataset_from_directory(
    os.path.join(base_dir, 'val'),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

test_ds_raw = image_dataset_from_directory(
    os.path.join(base_dir, 'test'),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Apply preprocessing
train_ds = prepare_dataset(train_ds_raw, training=True)
val_ds = prepare_dataset(val_ds_raw)
test_ds = prepare_dataset(test_ds_raw)

##Step -- 7
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

# Load MobileNetV2 base (pretrained on ImageNet)
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False  # Freeze base layers

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')  # 2 classes: PNEUMONIA, NORMAL
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Show summary
model.summary()

##Step -- 8
# Add early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)
import matplotlib.pyplot as plt
##Step -- 9
# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
##Step -- 10
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test Accuracy: {test_accuracy:.2f}")
##Step -- 11
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get true labels and predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_ds_raw.class_names,
            yticklabels=train_ds_raw.class_names)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=train_ds_raw.class_names))

##Step -- 12
# Save the entire model as a .h5 file
model.save("pneumonia_detection_model.h5")
print("✅ Model saved as 'pneumonia_detection_model.h5'")
## Step -- 13
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model("pneumonia_detection_model.h5")
class_names = ['NORMAL', 'PNEUMONIA']

# Title
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to predict if it shows signs of pneumonia.")

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show result
    st.markdown(f"### 🔍 Prediction: `{predicted_class}`")
    st.markdown(f"Confidence: **{confidence * 100:.2f}%**")
