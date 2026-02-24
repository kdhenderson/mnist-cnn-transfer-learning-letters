#!/usr/bin/env python3
# MNIST CNN Classifier for digits 0-9 (Full Dataset)
# Uses TensorFlow 2/Keras with a 3-layer CNN architecture
# This script trains a base model on all MNIST digits (0-9) for transfer learning

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Use the legacy Adam optimizer for Apple Silicon
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


def create_output_directories():
    """
    Creates output directories for figures and results at the project root.
    Returns:
        Tuple of (figures_dir, results_dir)
    """
    # Assume this file lives in src/ and go up one level to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(project_root, "figures")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return figures_dir, results_dir


def load_and_filter_mnist(digits_to_keep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    Load MNIST dataset and filter to keep only specified digits.
    Args:
        digits_to_keep: List of digits to keep in the dataset
    Returns:
        Tuple of filtered (x_train, y_train, x_test, y_test)
    """
    # Load the MNIST dataset
    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    # Create masks for the digits we want to keep
    train_mask = np.isin(y_train_full, digits_to_keep)
    test_mask = np.isin(y_test_full, digits_to_keep)
    # Filter the dataset
    x_train = x_train_full[train_mask]
    y_train = y_train_full[train_mask]
    x_test = x_test_full[test_mask]
    y_test = y_test_full[test_mask]
    # Remap labels to be consecutive (0-9 instead of 0,1,...,9)
    for i, digit in enumerate(digits_to_keep):
        y_train[y_train == digit] = i
        y_test[y_test == digit] = i
    return x_train, y_train, x_test, y_test


def create_custom_train_test_split(x, y, train_ratio=0.9):
    """
    Splits the dataset into training and testing sets.
    Args:
        x: Input data (e.g., images)
        y: Labels
        train_ratio: Proportion of data to use for training
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    # Ensure x and y have the same number of samples
    assert len(x) == len(y), "Input data and labels must have the same length."
    
    # Calculate the number of samples for training and testing
    total_samples = len(x)
    train_samples = int(total_samples * train_ratio)
    
    # Split the data
    x_train = x[:train_samples]
    y_train = y[:train_samples]
    x_test = x[train_samples:]
    y_test = y[train_samples:]
    
    return x_train, y_train, x_test, y_test


def preprocess_data(x_train, y_train, x_test, y_test, num_classes):
    """
    Preprocesses the data for training.
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        num_classes: Number of classes
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    # Add a channel dimension (for grayscale images)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def build_cnn_model(input_shape, num_classes):
    """
    Builds a CNN model for MNIST classification.
    Args:
        input_shape: Shape of input images (e.g., (28, 28, 1))
        num_classes: Number of classes
    Returns:
        Keras Sequential model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    """
    Trains the model and evaluates it on the test set.
    Args:
        model: Keras model
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        batch_size: Batch size for training
        epochs: Number of epochs
    Returns:
        Tuple of (trained_model, history)
    """
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Save checkpoints under the project-root results/ directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    best_model_path = os.path.join(project_root, 'results', 'best_model.h5')
    model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, model_checkpoint])

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    return model, history


def plot_training_history(history, model, figures_dir):
    """
    Plots the training and validation loss/accuracy history.
    Args:
        history: History object from model.fit
        model: Keras model
        figures_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    # Save with model name and timestamp
    figure_path = os.path.join(figures_dir, f"mnist_cnn_training_history_{timestamp}.png")
    plt.savefig(figure_path)
    plt.close()
    print(f"Learning curve plot saved to: {figure_path}")


def save_training_results(history, model, results_dir, x_train, x_test):
    """
    Saves training results to a JSON file.
    Args:
        history: History object from model.fit
        model: Keras model
        results_dir: Directory to save results
        x_train: Training images
        x_test: Test images
    """
    import json
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = {
        'model_architecture': model.to_json(),
        'training_history': history.history,
        'dataset_info': {
            'total_samples': len(x_train) + len(x_test),
            'train_samples': len(x_train),
            'test_samples': len(x_test)
        }
    }
    results_path = os.path.join(results_dir, f"training_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Training results saved to: {results_path}")


def main():
    """Main function to orchestrate the workflow."""
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    # Create output directories
    figures_dir, results_dir = create_output_directories()
    print(f"Output directories created: figures/ and results/")
    # Parameters
    digits_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_classes = len(digits_to_keep)
    train_ratio = 0.9
    batch_size = 64
    epochs = 50  # Increased epochs since we're using early stopping
    print("Loading and filtering MNIST dataset (digits 0-9)...")
    # Load and filter MNIST to keep all digits 0-9
    x_data, y_data, x_test_orig, y_test_orig = load_and_filter_mnist(digits_to_keep)
    print("Creating 90/10 train/test split...")
    # Combine original training and test data, then create a new 90/10 split
    x_combined = np.vstack([x_data, x_test_orig])
    y_combined = np.hstack([y_data, y_test_orig])
    x_train, y_train, x_test, y_test = create_custom_train_test_split(
        x_combined, y_combined, train_ratio
    )
    print("Preprocessing data...")
    # Preprocess the data
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test, num_classes
    )
    print(f"Training data shape: {x_train.shape}, Test data shape: {x_test.shape}")
    print("Building CNN model...")
    # Build the CNN model
    model = build_cnn_model((28, 28, 1), num_classes)
    model.summary()
    print("Training model...")
    # Train and evaluate the model
    model, history = train_and_evaluate_model(
        model, x_train, y_train, x_test, y_test, batch_size, epochs
    )
    # Print final training accuracy
    final_train_acc = history.history['accuracy'][-1]
    print(f"Final training accuracy: {final_train_acc:.4f}")
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, model, figures_dir)
    # Save training results to JSON
    print("Saving training results...")
    save_training_results(history, model, results_dir, x_train, x_test)
    # Save the model in a timestamped subdirectory
    import datetime
    model_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(results_dir, f"mnist_cnn_model_{model_timestamp}")
    model.save(model_dir)
    print(f"Model saved to: {model_dir}")
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total MNIST digits 0-9: {len(x_combined)}")
    print(f"Training set size: {len(x_train)} images")
    print(f"Test set size: {len(x_test)} images")


if __name__ == "__main__":
    main() 