# MNIST CNN Classifier with Transfer Learning

[Project Report](docs/project_report.md)

This project implements a Convolutional Neural Network (CNN) using TensorFlow 2/Keras to classify MNIST digits, with two main components:

1. A base CNN model trained on digits 0-4
2. A transfer learning model that reuses the base model to classify digits 5-9

## Features

### Base Model (`src/mnist_cnn.py`)
- Loads and filters the MNIST dataset to include only digits 0-4
- Creates a custom 90/10 train/test split
- Implements a CNN with 3 convolutional layers
- Uses ReLU activation in hidden layers and softmax for 5-class classification
- Includes proper preprocessing and evaluation

### Transfer Learning Model (`src/mnist_transfer_learning_tuned.py`)
- Loads the pre-trained model from the base CNN
- Freezes the convolutional and hidden layers
- Replaces the output layer for classifying digits 5-9
- Retrains only the new output layer
- Demonstrates how features learned from one set of digits can be applied to another

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Setup

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. First, train the base model on digits 0-4:

```bash
python src/mnist_cnn.py
```

2. Then, apply transfer learning to classify digits 5-9:

```bash
python src/mnist_transfer_learning_tuned.py
```

## Model Architecture

### Base CNN Model
- 3 convolutional layers with ReLU activation
- MaxPooling and Dropout after each convolutional layer
- A fully connected layer with ReLU activation
- Output layer with softmax activation for 5-class classification

### Transfer Learning Model
- Reuses the frozen feature extraction layers from the base model
- Replaces only the final classification layer
- Trains the new layer while keeping pre-trained weights fixed

## Output and Visualization

Both scripts will:
- Save learning curve plots to the `figures/` directory
- Save training metrics and model summary to JSON files in the `results/` directory
- Save the trained models to the `results/` directory
- Print progress during training, final training accuracy, and test accuracy 

---

## Extension: Transfer Learning from Full MNIST (0-9) to Handwritten Letters (A–E)

This section describes an additional experiment that extends the original workflow. Here, the base CNN is trained on the full MNIST dataset (digits 0–9), and transfer learning is applied to a custom dataset of handwritten letters (A–E) that have been digitized and preprocessed to match MNIST format.

### Motivation
- The architecture and transfer learning approach were first developed and tuned on the original 0–4/5–9 digit split (see above).
- This extension demonstrates how the same approach can be used to adapt a digit-trained model to a new domain: handwritten letters.

### New Scripts
- **`src/train_mnist_base.py`**: Trains a CNN on all MNIST digits (0–9) using the same architecture as the original base model. The resulting model is saved for transfer learning.
- **`src/transfer_to_letters.py`**: Loads the base model, freezes the convolutional layers, and adapts the model for 5-class classification (A–E). Trains on a custom dataset of processed handwritten letters. Supports training with or without data augmentation via the `--augment` flag.

### Handwritten Letters Dataset
- **Location:** `data/processed_letters/set3/`
- **Structure:**
  - Subdirectories `A/`, `B/`, `C/`, `D/`, `E/` each contain 27 images of the corresponding letter.
  - Images are 28x28 grayscale PNGs, binarized and centered to match MNIST.
- **Train/Test Split:** 22 images per class for training, 5 per class for testing (stratified).

### How to Run

1. **Train the base model on all MNIST digits (0–9):**
   ```bash
   python src/train_mnist_base.py
   ```
   - Model and training results are saved in the `results/` directory.

2. **Transfer learning to handwritten letters (A–E):**
   - **Without augmentation (baseline):**
     ```bash
     python src/transfer_to_letters.py
     ```
   - **With augmentation:**
     ```bash
     python src/transfer_to_letters.py --augment
     # or
     python src/transfer_to_letters.py -a
     ```
   - The `--augment` flag enables random rotation, translation, and zoom during training.
   - All results and models are saved with `noaug` or `aug` in the filename for easy comparison.

### Output and Results
- **figures/**: Contains training history plots and confusion matrices for both modes (with and without augmentation), with clear prefixes and timestamps.
- **results/**: Contains the saved models and JSON files with training metrics, also with clear prefixes and timestamps.

### Notes
- This extension demonstrates the flexibility of transfer learning: a model trained on digits can be quickly adapted to new handwritten character classes with minimal data, especially when preprocessing is consistent.
- The architecture and training strategy were first validated on the 0–4/5–9 digit split, then applied to the more challenging letter classification task.
- You can compare the effect of data augmentation by running the transfer learning script with and without the `--augment` flag and reviewing the saved results and plots. 

---

## Output Files and Naming Conventions

The following table summarizes how each script names and saves its outputs. All outputs are timestamped for reproducibility and easy comparison.

| Script                        | Figures (plots)                                   | Results JSON                                    | Model Directory (SavedModel)                      |
|-------------------------------|---------------------------------------------------|--------------------------------------------------|---------------------------------------------------|
| **src/mnist_cnn.py**              | `figures/training_history_<timestamp>.png`        | `results/training_results_<timestamp>.json`      | `results/mnist_cnn_model_<timestamp>/`            |
| **src/mnist_transfer_learning_tuned.py** | `figures/transfer_tuned_training_history_<timestamp>.png` | `results/transfer_tuned_training_results_<timestamp>.json` | `results/mnist_transfer_tuned_model_<timestamp>/` |
| **src/train_mnist_base.py**       | `figures/mnist_cnn_training_history_<timestamp>.png` | `results/training_results_<timestamp>.json`      | `results/mnist_cnn_model_<timestamp>/`            |
| **src/transfer_to_letters.py**    | `figures/transfer_letters_noaug_training_history_<timestamp>.png`<br>`figures/transfer_letters_aug_training_history_<timestamp>.png`<br>`figures/transfer_letters_noaug_confusion_matrix_<timestamp>.png`<br>`figures/transfer_letters_aug_confusion_matrix_<timestamp>.png` | `results/transfer_letters_noaug_training_results_<timestamp>.json`<br>`results/transfer_letters_aug_training_results_<timestamp>.json` | `results/mnist_transfer_letters_model_transfer_letters_noaug_<timestamp>/`<br>`results/mnist_transfer_letters_model_transfer_letters_aug_<timestamp>/` |

- `<timestamp>` is the date and time the script was run, ensuring unique and chronological output files.
- All figures are saved in the `figures/` directory.
- All results JSON files and model directories are saved in the `results/` directory.
- For `src/transfer_to_letters.py`, the prefix `noaug` or `aug` indicates whether data augmentation was used.

Refer to this table to quickly locate the outputs from each experiment for analysis or reporting. 

---

## Additional Documentation

- [`docs/project_report.md`](docs/project_report.md): Detailed write-up of experiments and results.
- [`docs/assignment_description.md`](docs/assignment_description.md): Original homework instructions.
- [`docs/LETTERS_README.md`](docs/LETTERS_README.md): Guide to the letter-processing scripts and dataset.
- [`docs/IMAGE_PROCESSING_TECHNIQUES.md`](docs/IMAGE_PROCESSING_TECHNIQUES.md): Image processing and OpenCV techniques primer.
- [`docs/transfer_learning_improvements.md`](docs/transfer_learning_improvements.md): Explanation of transfer learning model tuning.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.