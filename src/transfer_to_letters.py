#!/usr/bin/env python3
"""
Transfer Learning: Adapt MNIST CNN to Handwritten Letters (A-E)
Stage 1: Data Loading, Labeling, and Visualization

- Loads all images from processed_letters/set3/A-E
- Assigns numeric labels (A=0, B=1, ..., E=4)
- Splits into 22 train / 5 test per class (stratified)
- Visualizes a few samples per class
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import datetime
import json
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

# Map letter to numeric label
def letter_to_label(letter):
    return ord(letter.upper()) - ord('A')

# Map numeric label to letter
def label_to_letter(label):
    return chr(label + ord('A'))

# Directory containing letter images
DATA_DIR = os.path.join('data', 'processed_letters', 'set3')
CLASSES = ['A', 'B', 'C', 'D', 'E']
IMAGES_PER_CLASS = 27
TRAIN_PER_CLASS = 22
TEST_PER_CLASS = 5
IMG_SIZE = (28, 28)

# Load images and labels
def load_letter_images(data_dir, classes):
    images = []
    labels = []
    for letter in classes:
        class_dir = os.path.join(data_dir, letter)
        filenames = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
        for fname in filenames:
            img_path = os.path.join(class_dir, fname)
            img = Image.open(img_path).convert('L')  # Grayscale
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
            images.append(img_array)
            labels.append(letter_to_label(letter))
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Stratified split: 22 train, 5 test per class
def stratified_split(images, labels, train_per_class, test_per_class, n_classes):
    x_train, y_train, x_test, y_test = [], [], [], []
    for label in range(n_classes):
        idx = np.where(labels == label)[0]
        np.random.shuffle(idx)
        train_idx = idx[:train_per_class]
        test_idx = idx[train_per_class:train_per_class+test_per_class]
        x_train.append(images[train_idx])
        y_train.append(labels[train_idx])
        x_test.append(images[test_idx])
        y_test.append(labels[test_idx])
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    return x_train, y_train, x_test, y_test

# Visualize a few samples per class
def visualize_samples(images, labels, classes, samples_per_class=5):
    plt.figure(figsize=(10, 4))
    for i, label in enumerate(range(len(classes))):
        idx = np.where(labels == label)[0][:samples_per_class]
        for j, img_idx in enumerate(idx):
            plt.subplot(len(classes), samples_per_class, i * samples_per_class + j + 1)
            plt.imshow(images[img_idx], cmap='gray')
            plt.axis('off')
            if j == 0:
                plt.ylabel(classes[i], fontsize=14)
    plt.suptitle('Sample Images per Class', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- Stage 2: Data Augmentation and Visualization ---
def build_augmentation_layer():
    """
    Create a Keras Sequential model with augmentation layers:
    - Random rotation (±10 degrees)
    - Random translation (10% height/width)
    - Random zoom (±10%)
    No flipping or shearing is applied (shearing is not supported by the augmentation layer)
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03, fill_mode='nearest'),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='nearest'),
        tf.keras.layers.RandomZoom(0.1, 0.1, fill_mode='nearest'),
    ], name='augmentation')

def visualize_augmented_samples(x_train, y_train, classes, augmentation_layer, samples_per_class=5):
    """
    Visualize original and augmented images for each class.
    """
    plt.figure(figsize=(12, 6))
    for i, label in enumerate(range(len(classes))):
        idx = np.where(y_train == label)[0][:samples_per_class]
        for j, img_idx in enumerate(idx):
            # Original
            plt.subplot(2 * len(classes), samples_per_class, i * samples_per_class + j + 1)
            plt.imshow(x_train[img_idx].squeeze(), cmap='gray')
            plt.axis('off')
            if j == 0:
                plt.ylabel(f"{classes[i]}\nOriginal", fontsize=12)
            # Augmented
            aug_img = augmentation_layer(x_train[img_idx][np.newaxis, ...], training=True)
            aug_img = tf.squeeze(aug_img).numpy()
            plt.subplot(2 * len(classes), samples_per_class, (i + len(classes)) * samples_per_class + j + 1)
            plt.imshow(aug_img, cmap='gray')
            plt.axis('off')
            if j == 0:
                plt.ylabel(f"{classes[i]}\nAugmented", fontsize=12)
    plt.suptitle('Original and Augmented Images per Class', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- Stage 3: Model Adaptation and Training ---
def find_latest_model(results_dir, prefix="mnist_cnn_model_"):
    """
    Find the most recently saved base model directory in results_dir.
    """
    model_dirs = [d for d in os.listdir(results_dir) if d.startswith(prefix)]
    if not model_dirs:
        return None
    latest_model = sorted(model_dirs)[-1]
    return os.path.join(results_dir, latest_model)

def build_transfer_model(base_model, num_classes):
    """
    Adapt the base model for transfer learning:
    - Freeze all convolutional layers
    - Add robust dense layers (as in mnist_transfer_learning_tuned.py)
    - Output layer for num_classes
    """
    transfer_model = models.Sequential()
    # Add and freeze all layers up to Flatten
    for layer in base_model.layers:
        if isinstance(layer, layers.Flatten):
            transfer_model.add(layer)
            break
        layer.trainable = False
        transfer_model.add(layer)
    # Add robust dense layers (copied from tuned script)
    transfer_model.add(layers.Dense(256, activation='relu', name='transfer_dense_1'))
    transfer_model.add(layers.Dropout(0.3, name='transfer_dropout_1'))
    transfer_model.add(layers.Dense(128, activation='relu', name='transfer_dense_2'))
    transfer_model.add(layers.Dropout(0.2, name='transfer_dropout_2'))
    transfer_model.add(layers.Dense(num_classes, activation='softmax', name='output_letters'))
    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return transfer_model

def plot_training_history(history, figures_dir, prefix="transfer_letters_"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    figure_path = os.path.join(figures_dir, f'{prefix}training_history_{timestamp}.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Learning curve plot saved to: {figure_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, figures_dir, prefix="transfer_letters_"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, f'{prefix}confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    plt.show()

def save_training_results(history, model, results_dir, x_train, x_test, prefix="transfer_letters_"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
    model_summary_list = []
    model.summary(print_fn=lambda x: model_summary_list.append(x))
    results['model_summary'] = model_summary_list
    results_path = os.path.join(results_dir, f'{prefix}training_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Training results saved to: {results_path}")

# --- Main logic update ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer learning for handwritten letters (A-E)')
    parser.add_argument('--augment', '-a', action='store_true', help='Use data augmentation during training')
    args = parser.parse_args()
    np.random.seed(42)
    print("Loading letter images from:", DATA_DIR)
    images, labels = load_letter_images(DATA_DIR, CLASSES)
    print(f"Loaded {len(images)} images. Shape: {images.shape}")
    print("Performing stratified train/test split (22 train, 5 test per class)...")
    x_train, y_train, x_test, y_test = stratified_split(images, labels, TRAIN_PER_CLASS, TEST_PER_CLASS, len(CLASSES))
    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Test set: {x_test.shape}, {y_test.shape}")
    print("Visualizing samples...")
    visualize_samples(images, labels, CLASSES, samples_per_class=5)
    # Stage 2: Data Augmentation
    print("\nBuilding augmentation pipeline and visualizing augmented samples...")
    augmentation_layer = build_augmentation_layer()
    x_train_aug = x_train[..., np.newaxis]
    visualize_augmented_samples(x_train_aug, y_train, CLASSES, augmentation_layer, samples_per_class=5)
    print("Stage 2 complete. Ready for model adaptation and training.")
    # One-hot encode labels for training
    y_train_cat = to_categorical(y_train, num_classes=len(CLASSES))
    y_test_cat = to_categorical(y_test, num_classes=len(CLASSES))
    # Load latest base model
    results_dir = 'results'
    print("\nFinding latest base MNIST model...")
    latest_model_path = find_latest_model(results_dir)
    if not latest_model_path:
        print("Error: No pre-trained model found in the results directory.")
        exit(1)
    print(f"Found pre-trained model: {latest_model_path}")
    base_model = load_model(latest_model_path)
    # Build transfer model
    print("Building transfer learning model for letters A-E...")
    transfer_model = build_transfer_model(base_model, num_classes=len(CLASSES))
    transfer_model.summary()
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    batch_size = 16
    epochs = 50
    # Choose training mode based on --augment flag
    if args.augment:
        print("\nTraining with data augmentation...")
        prefix = "transfer_letters_aug_"
        # Use augmentation layer in a tf.data pipeline
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train[..., np.newaxis], y_train_cat))
        train_dataset = train_dataset.shuffle(buffer_size=100, seed=42).batch(batch_size)
        train_dataset = train_dataset.map(lambda x, y: (augmentation_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        print("\nTraining WITHOUT data augmentation...")
        prefix = "transfer_letters_noaug_"
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train[..., np.newaxis], y_train_cat))
        train_dataset = train_dataset.shuffle(buffer_size=100, seed=42).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test[..., np.newaxis], y_test_cat)).batch(batch_size)
    # Train
    history = transfer_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = transfer_model.evaluate(val_dataset, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    # Predictions for confusion matrix
    y_pred_probs = transfer_model.predict(val_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(np.vstack([y for x, y in val_dataset]), axis=1)
    # Plot training history
    plot_training_history(history, 'figures', prefix=prefix)
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, CLASSES, 'figures', prefix=prefix)
    # Save training results
    print("Saving training results JSON...")
    save_training_results(history, transfer_model, 'results', x_train, x_test, prefix=prefix)
    print("Training results JSON saved.")
    # Save the fine-tuned model
    print("Saving fine-tuned transfer learning model...")
    model_path = os.path.join('results', f"mnist_transfer_letters_model_{prefix}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    transfer_model.save(model_path)
    print(f"Fine-tuned transfer learning model saved to: {model_path}")
    print("\nAll done!") 