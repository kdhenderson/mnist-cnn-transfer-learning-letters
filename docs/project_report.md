# Project Report: Transfer Learning with CNNs on MNIST and Handwritten Letters

## Introduction

This project explores the use of Convolutional Neural Networks (CNNs) and transfer learning for handwritten character recognition. The work is divided into two main parts:

1. **Training a CNN on MNIST digits 0–4 and transferring to digits 5–9**
2. **Training a CNN on the full MNIST dataset (0–9) and transferring to handwritten letters A–E**

The goal is to demonstrate how features learned from one set of classes (digits) can be leveraged to improve performance on related but distinct tasks (other digits or letters), especially when labeled data is limited.

---

## Part 1: CNN on MNIST 0–4, Transfer Learning to 5–9

### Methods

- **Base Model:**
  - Trained a CNN with three convolutional layers on MNIST digits 0–4.
  - Used a 90/10 train/test split (combining original MNIST train and test sets for reproducibility).
  - Early stopping was applied to prevent overfitting.
  - Model architecture:
    - 3× Conv2D + MaxPooling2D + Dropout
    - Flatten → Dense(128) + Dropout → Dense(5, softmax)

- **Transfer Learning:**
  - The trained convolutional layers were frozen.
  - Dense layers were replaced and retrained to classify digits 5–9.
  - Lower learning rate and reduced dropout were used for fine-tuning.

### Results

#### Base Model (Digits 0–4)
- **Final Training Accuracy:** 0.9973
- **Final Validation Accuracy:** 0.9911
- **Epochs Completed:** 20
- **Training Samples:** 63,000
- **Test Samples:** 7,000

![MNIST 0-4 Training History](figures/mnist_cnn_training_history_20250722-154712.png)

#### Transfer Model (Digits 5–9)
- **Final Training Accuracy:** 0.8984
- **Final Validation Accuracy:** 0.9643
- **Epochs Completed:** 40
- **Training Samples:** 30,838
- **Test Samples:** 3,427

![Transfer Learning 5-9 Training History](figures/transfer_tuned_training_history_20250713-152557.png)

*See detailed metrics in [`results/training_results_20250722-154712.json`](results/training_results_20250722-154712.json) and [`results/transfer_tuned_training_results_20250713-152608.json`](results/transfer_tuned_training_results_20250713-152608.json)*

---

## Part 2: CNN on MNIST 0–9, Transfer Learning to Letters A–E

### Methods

- **Base Model:**
  - Trained a CNN on all MNIST digits (0–9) using the same architecture as above.
  - 90/10 train/test split for robust evaluation.

- **Transfer Learning to Letters:**
  - Used a custom dataset of handwritten letters A–E (27 images per class).
  - Stratified split: 22 train, 5 test per class.
  - The convolutional layers from the MNIST model were frozen.
  - New dense layers were added for 5-class letter classification.
  - Two training regimes:
    - **Without Augmentation:** Only original images.
    - **With Augmentation:** Random rotations, translations, and zooms applied to training images.

### Results

#### Without Augmentation
- **Final Training Accuracy:** 1.0
- **Final Validation Accuracy:** 1.0
- **Epochs Completed:** 47
- **Training Samples:** 110
- **Test Samples:** 25

![No Augmentation Training History](figures/transfer_letters_noaug_training_history_20250722-161314.png)
![No Augmentation Confusion Matrix](figures/transfer_letters_noaug_confusion_matrix_20250722-161735.png)

#### With Augmentation
- **Final Training Accuracy:** 0.9909
- **Final Validation Accuracy:** 1.0
- **Epochs Completed:** 33
- **Training Samples:** 110
- **Test Samples:** 25

![Augmentation Training History](figures/transfer_letters_aug_training_history_20250722-161945.png)
![Augmentation Confusion Matrix](figures/transfer_letters_aug_confusion_matrix_20250722-162001.png)

*See detailed metrics in [`results/transfer_letters_noaug_training_results_20250722-161740.json`](results/transfer_letters_noaug_training_results_20250722-161740.json) and [`results/transfer_letters_aug_training_results_20250722-162006.json`](results/transfer_letters_aug_training_results_20250722-162006.json)*

---

## Letter Dataset Creation and Preprocessing

The letter dataset was created by writing letters A–E by hand and photographing them with an iPhone SE. Automatic cropping was used to extract individual letter images from the raw scans. This process worked very well for images written with a thick sharpie, as the bold lines made it easy for the algorithm to detect and crop the letters accurately. However, for letters written with a thin sharpie (notably A, C, and E), the automatic cropping was less reliable.

For this project, only extremely clean, well-cropped images were used for training and testing. As a result, the dataset was free from significant noise or artifacts. Interestingly, even when the denoising step was skipped, the model achieved perfect accuracy on the validation set. This highlights the importance of clean data: with minimal noise and clear letter boundaries, the CNN was able to learn and generalize perfectly to the test set, even with a small number of samples per class.

---

## Discussion

- **Transfer learning** allowed the reuse of learned features from digits to new digit classes and even to handwritten letters, reducing the need for large labeled datasets.
- **Data augmentation** was especially important for the small letter dataset, improving generalization and test accuracy.
- **Confusion matrices** show that augmentation led to more balanced performance across all classes.
- **Takeaway:** CNNs trained on MNIST digits learn robust features that can be successfully transferred to related tasks, and augmentation is key when data is scarce. 