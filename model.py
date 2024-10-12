import os
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load CSV for class information
labels_df = pd.read_csv('../dataset/train/_classes.csv')

# Strip whitespace from column headers
labels_df.columns = labels_df.columns.str.strip()

# Create a new column 'label' to represent the class (Fresh, Half-Fresh, Spoiled)
# Assumes that each image belongs to exactly one class
def get_label(row):
    if row['Fresh'] == 1:
        return 'Fresh'
    elif row['Half-Fresh'] == 1:
        return 'Half-Fresh'
    else:
        return 'Spoiled'

labels_df['label'] = labels_df.apply(get_label, axis=1)

# Define data directories
train_data_dir = '../dataset/train'

# Preprocess images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create the training data generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=train_data_dir,
    x_col='filename', 
    y_col='label',  # Use the 'label' column created above
    target_size=(416, 416),
    batch_size=32,
    class_mode='categorical',  # Keras will one-hot encode this internally
    shuffle=True  # Shuffle data during training
)

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(416, 416, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Output layer with 3 units (for 3 classes)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model 
model.fit(
    train_generator,
    epochs=10, 
) 

# Save the trained model
model.save('model.h5')
