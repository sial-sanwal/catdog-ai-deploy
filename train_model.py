"""
CatDog Vision AI - Model Training Script
Trains a CNN model to classify cats and dogs
"""

import os
import zipfile
import urllib.request
import tempfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Constants
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "cats_and_dogs_filtered"
MODEL_PATH = "catdog_model.h5"

def download_dataset():
    """Download and extract the cats and dogs dataset"""
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    
    if os.path.exists(DATA_DIR):
        print("Dataset already exists, skipping download...")
        return
    
    # Create temporary file for download
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, "cats_and_dogs_filtered.zip")
    
    print("Downloading dataset...")
    print(f"This may take a few minutes. Downloading from: {dataset_url}")
    
    try:
        # Download the file
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Download complete!")
        
        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Dataset extracted successfully!")
        
        # Clean up zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nYou can manually download the dataset from:")
        print(dataset_url)
        print("And extract it to the current directory.")
        raise

def create_model():
    """Create CNN model architecture"""
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Fourth convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the CNN model"""
    print("Setting up data generators...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data (only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    # Validation generator
    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'validation'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    
    # Create model
    print("\nCreating CNN model...")
    model = create_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.0001
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(
        validation_generator,
        steps=validation_generator.samples // BATCH_SIZE
    )
    
    print(f"\nFinal Validation Accuracy: {test_accuracy:.2%}")
    print(f"Final Validation Loss: {test_loss:.4f}")
    
    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Model saved successfully!")
    
    return model, history

def main():
    """Main function"""
    print("=" * 50)
    print("CatDog Vision AI - Model Training")
    print("=" * 50)
    
    # Download dataset if needed
    download_dataset()
    
    # Train model
    model, history = train_model()
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()

