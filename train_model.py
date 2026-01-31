"""
Training Script for Colorectal Cell Classifier
Trains the EfficientNetB0 model and saves it for use in the Streamlit app
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request
from pathlib import Path

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.image import resize_with_pad
from sklearn.model_selection import train_test_split

# Set project
project = "histology"

def download_data(project, download_url_prefix_dict):
    """Download data files if they don't already exist locally.
    
    Args:
        project: Project name (e.g., 'histology')
        download_url_prefix_dict: Dictionary mapping project names to download URLs
        
    Returns:
        Tuple of (images_path, labels_path) as strings
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    images_path = data_dir / "images.npy"
    labels_path = data_dir / "labels.npy"
    
    if images_path.exists() and labels_path.exists():
        print("Data files already exist, skipping download...")
        return str(images_path), str(labels_path)
    
    base_url = download_url_prefix_dict[project]
    images_url = base_url + "images.npy"
    labels_url = base_url + "labels.npy"
    
    print(f"Downloading images from {images_url}...")
    print("This may take a few minutes (~500MB)...")
    urllib.request.urlretrieve(images_url, images_path)
    print(f"Downloaded images to {images_path}")
    
    print(f"Downloading labels from {labels_url}...")
    urllib.request.urlretrieve(labels_url, labels_path)
    print(f"Downloaded labels to {labels_path}")
    
    return str(images_path), str(labels_path)

def ResizeImages(images, height, width):
    """Resize images to target dimensions using TensorFlow's resize_with_pad."""
    return np.array([resize_with_pad(image, height, width, antialias=True) for image in images]).astype(int)

def plot_training_history(history):
    """Plot training and validation accuracy/loss over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig

def main():
    print("=" * 60)
    print("Colorectal Cell Classifier - Training Script")
    print("=" * 60)
    print()
    
    # Download URL dictionary
    download_url_prefix_dict = {
        "histology": "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Towards%20Precision%20Medicine/",
    }
    
    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    images_path, labels_path = download_data(project, download_url_prefix_dict)
    images = np.load(images_path)
    labels = np.load(labels_path)
    print(f"Loaded {len(images)} images with {len(set(labels))} unique labels")
    print(f"Labels: {sorted(set(labels))}")
    print()
    
    # Step 2: Prepare data
    print("Step 2: Preparing data...")
    # One-hot encode labels
    labels_ohe = np.array(pd.get_dummies(labels))
    
    # Select features and labels
    y = labels_ohe
    X = images
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()
    
    # Step 3: Resize images
    print("Step 3: Resizing images to 224x224...")
    X_train_resized = ResizeImages(X_train, 224, 224)
    X_test_resized = ResizeImages(X_test, 224, 224)
    print("Images resized successfully")
    print()
    
    # Step 4: Build model
    print("Step 4: Building model...")
    # Load EfficientNetB0 base model
    efficientNet_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
    
    # Freeze all layers
    for layer in efficientNet_model.layers:
        layer.trainable = False
    
    # Add custom layers
    x = efficientNet_model.output
    x = Flatten()(x)
    output = Dense(8, activation="softmax")(x)
    
    # Create final model
    transfer_cnn = Model(inputs=efficientNet_model.input, outputs=output)
    
    # Make only the new dense layer trainable (already true, but explicitly set)
    transfer_cnn.layers[-1].trainable = True
    
    # Compile model
    transfer_cnn.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "categorical_crossentropy"]
    )
    
    print("Model built successfully!")
    print(f"Total parameters: {transfer_cnn.count_params():,}")
    print()
    
    # Step 5: Train model
    print("Step 5: Training model...")
    print("This may take several minutes (5 epochs)...")
    print("-" * 60)
    
    history = transfer_cnn.fit(
        X_train_resized, y_train,
        epochs=5,
        validation_data=(X_test_resized, y_test),
        verbose=1
    )
    
    print("-" * 60)
    print("Training completed!")
    print()
    
    # Step 6: Evaluate model
    print("Step 6: Evaluating model...")
    train_loss, train_acc, train_ce = transfer_cnn.evaluate(X_train_resized, y_train, verbose=0)
    test_loss, test_acc, test_ce = transfer_cnn.evaluate(X_test_resized, y_test, verbose=0)
    
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    print()
    
    # Step 7: Save model
    print("Step 7: Saving model...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "trained_model.weights.h5"
    transfer_cnn.save_weights(str(model_path))
    print(f"Model weights saved to: {model_path}")
    print()
    
    # Step 8: Save training history plot
    print("Step 8: Saving training history plot...")
    fig = plot_training_history(history)
    plot_path = models_dir / "training_history.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to: {plot_path}")
    plt.close(fig)
    print()
    
    print("=" * 60)
    print("Training script completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. The trained model is saved at: {model_path}")
    print("  2. You can now run the Streamlit app: streamlit run app.py")
    print("  3. The app will automatically load the trained model")
    print()

if __name__ == "__main__":
    main()
