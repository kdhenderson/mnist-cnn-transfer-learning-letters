#!/usr/bin/env python3
# MNIST CNN Classifier for digits 0-4
# Uses TensorFlow 2/Keras with a 3-layer CNN architecture

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import datetime
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

def load_and_filter_mnist(digits_to_keep=[0, 1, 2, 3, 4]):
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
    
    # Remap labels to be consecutive (0-4 instead of 0,1,2,3,4)
    # This ensures class labels are 0-4 regardless of which digits we keep
    # Essential for proper one-hot encoding and softmax output interpretation
    for i, digit in enumerate(digits_to_keep):
        y_train[y_train == digit] = i
        y_test[y_test == digit] = i
    
    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, y_train, x_test, y_test, num_classes):
    """
    Preprocess the data for CNN training:
    - Normalize pixel values to [0,1]
    - Reshape to include channel dimension
    - Convert labels to categorical format
    
    Args:
        x_train, y_train, x_test, y_test: Training and test data
        num_classes: Number of classes for categorical conversion
        
    Returns:
        Preprocessed data ready for model training
    """
    # Normalize pixel values to [0,1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to include channel dimension (height, width, channels)
    # CNN layers in Keras/TensorFlow require 4D input: (batch_size, height, width, channels)
    # Even though MNIST is grayscale, we need to add the channel dimension (1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert labels to categorical one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test

def create_custom_train_test_split(x, y, train_ratio=0.9):
    """
    Create a custom train/test split with the specified ratio.
    
    Note: In practice, sklearn's train_test_split would be preferred:
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    
    Args:
        x, y: Data and labels
        train_ratio: Proportion of data to use for training
        
    Returns:
        x_train, y_train, x_test, y_test
    """
    # Calculate the split index
    split_idx = int(len(x) * train_ratio)
    
    # Create shuffled indices
    indices = np.random.permutation(len(x))
    
    # Split the data
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    
    return x_train, y_train, x_test, y_test

def build_cnn_model(input_shape, num_classes):
    """
    Build a CNN model with at least 3 convolutional layers.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential()
    
    # First convolutional layer - detects basic features like edges and lines
    # 32 filters provide enough capacity to capture simple patterns
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))  # Reduces spatial dimensions, adds translation invariance
    model.add(layers.Dropout(0.25))  # Light dropout to prevent overfitting
    
    # Second convolutional layer - combines simple features into more complex patterns
    # Increased filters (64) to capture more complex feature combinations
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Third convolutional layer - identifies higher-level features specific to digits
    # Further increased filters (128) for more complex feature representation
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))  # Increased dropout as we go deeper
    
    # Flatten layer - converts 2D feature maps to 1D vector for dense layers
    model.add(layers.Flatten())
    
    # Dense layer - learns high-level combinations of features for classification
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Standard 0.5 dropout provides optimal regularization before classification
    
    # Output layer - softmax activation for multi-class classification
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',  # Adaptive learning rate optimizer
        loss='categorical_crossentropy',  # Standard loss function for multi-class classification
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=50):
    """
    Train the model and evaluate its performance.
    
    Args:
        model: Compiled Keras model
        x_train, y_train, x_test, y_test: Training and test data
        batch_size: Batch size for training
        epochs: Number of training epochs (higher value with early stopping)
        
    Returns:
        Trained model and training history
    """
    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',      # Monitor validation loss
        patience=5,              # Wait for 5 epochs without improvement
        restore_best_weights=True,  # Restore weights from best epoch
        verbose=1                # Print messages when stopping early
    )
    
    # Train the model with early stopping
    # validation_split=0.1 creates a separate validation set (different from test set) for early stopping
    # This creates 3 sets: training (81%), validation (9%), and test (10%)
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    return model, history

def create_output_directories():
    """
    Create directories for saving outputs if they don't exist.
    
    Returns:
        Tuple of (figures_dir, results_dir)
    """
    # Create directories if they don't exist
    figures_dir = os.path.join(os.getcwd(), 'figures')
    results_dir = os.path.join(os.getcwd(), 'results')
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return figures_dir, results_dir

def plot_training_history(history, model, figures_dir):
    """
    Plot training & validation loss and accuracy values and save to figures directory.
    
    Args:
        history: History object returned from model.fit()
        model: The trained model
        figures_dir: Directory to save figures
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    figure_path = os.path.join(figures_dir, f'training_history_{timestamp}.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Learning curve plot saved to: {figure_path}")
    
    plt.show()
    
def save_training_results(history, model, results_dir, x_train, x_test):
    """
    Save training results and model summary to a JSON file.
    
    Args:
        history: History object from model training
        model: The trained model
        results_dir: Directory to save results
        x_train: Training data to get shape information
        x_test: Test data to get shape information
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Extract relevant metrics
    results = {
        'timestamp': timestamp,
        'model_summary': [],
        'training_history': {
            'loss': [float(val) for val in history.history['loss']],
            'accuracy': [float(val) for val in history.history['accuracy']],
            'val_loss': [float(val) for val in history.history['val_loss']],
            'val_accuracy': [float(val) for val in history.history['val_accuracy']]
        },
        'final_metrics': {
            'train_loss': float(history.history['loss'][-1]),
            'train_accuracy': float(history.history['accuracy'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1])
        },
        'dataset_info': {
            'training_samples': int(x_train.shape[0]),
            'test_samples': int(x_test.shape[0]),
            'input_shape': [int(dim) for dim in x_train.shape[1:]]
        },
        'epochs_completed': len(history.history['loss'])
    }
    
    # Get model summary
    model_summary_list = []
    model.summary(print_fn=lambda x: model_summary_list.append(x))
    results['model_summary'] = model_summary_list
    
    # Save to file
    results_path = os.path.join(results_dir, f'training_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
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
    digits_to_keep = [0, 1, 2, 3, 4]
    num_classes = len(digits_to_keep)
    train_ratio = 0.9
    batch_size = 64
    epochs = 50  # Increased epochs since we're using early stopping
    
    print("Loading and filtering MNIST dataset...")
    # Load and filter MNIST to keep only digits 0-4
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
    
    # Save the model
    model_path = os.path.join(results_dir, f"mnist_cnn_model_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total MNIST digits 0-4: {len(x_combined)}")
    print(f"Training set size: {len(x_train)} images")
    print(f"Test set size: {len(x_test)} images")

if __name__ == "__main__":
    main() 