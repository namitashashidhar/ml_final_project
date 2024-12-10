import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
from PIL import Image

# Load dataset
data = pd.read_csv('train.csv')

# List of genre columns (excluding 'Id' and 'Genre' as they are not binary labels)
GENRE_COLUMNS = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery",
    "N/A", "News", "Reality-TV", "Romance", "Sci-Fi", "Short", "Sport", "Thriller", "War", "Western"
]

# Extract labels as a numpy array
labels = data[GENRE_COLUMNS].values
num_classes = labels.shape[1]

# Image size and batch parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Function to load and preprocess images
def load_image(img_name):
    img_path = os.path.join('Images', img_name + '.jpg')  # Updated to use "Images" folder
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    return img

# Load images and labels into arrays
images = np.array([load_image(img_name) for img_name in data['Id']])
labels = np.array(labels)

# Split data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Image Data Generator for augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator()  # No augmentation for validation data

# Build the CNN model with ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Unfreeze the last few layers for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='sigmoid')  # Sigmoid for multi-label classification
])

# Compile model with a lower learning rate
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=val_datagen.flow(X_val, y_val),
    epochs=20,
    callbacks=[early_stopping]
)

# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

# Save the model
model.save('movie_genre_classifier_model.keras')
