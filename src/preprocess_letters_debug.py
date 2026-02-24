#!/usr/bin/env python3
# Preprocessing script for handwritten letter images
# This script processes images of handwritten letters, extracts individual letters,
# and prepares them for use in a transfer learning model

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import argparse

def create_directories(base_path="data/processed_letters"):
    """
    Create directory structure for processed images.
    
    Args:
        base_path: Base directory for processed images
        
    Returns:
        Dictionary of paths for different letter categories
    """
    # Create main directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Create subdirectories for each letter (A-E)
    letter_dirs = {}
    for letter in "ABCDE":
        letter_path = os.path.join(base_path, letter)
        os.makedirs(letter_path, exist_ok=True)
        letter_dirs[letter] = letter_path
        
    # Create a directory for debugging/visualization
    debug_path = os.path.join(base_path, "debug")
    os.makedirs(debug_path, exist_ok=True)
    
    return letter_dirs, debug_path

def load_image(image_path):
    """
    Load an image from path and convert to grayscale.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (original image, grayscale image)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB for visualization (OpenCV uses BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image_rgb, gray

def deskew_image(image_rgb, gray_image):
    """
    Deskew the image to correct for rotation.
    
    Args:
        image_rgb: Original RGB image
        gray_image: Grayscale image
        
    Returns:
        Tuple of (deskewed RGB image, deskewed grayscale image)
    """
    # Find coordinates of non-white pixels
    coords = np.column_stack(np.where(gray_image < 240))
    
    # If no dark pixels found, return original
    if len(coords) < 10:
        return image_rgb, gray_image
    
    # Find minimum area rectangle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    print(f"Deskewing image by {angle:.2f} degrees")
    
    # Get image dimensions
    h, w = image_rgb.shape[:2]
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    
    # Apply rotation
    deskewed_rgb = cv2.warpAffine(
        image_rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    deskewed_gray = cv2.warpAffine(
        gray_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    
    return deskewed_rgb, deskewed_gray

def detect_and_crop_paper(image_rgb, gray_image, debug_path=None, image_name=None):
    """
    Detect the paper/document in the image and crop to it.
    
    Args:
        image_rgb: Original RGB image
        gray_image: Grayscale image
        debug_path: Path to save debug images
        image_name: Name of the image for debug purposes
        
    Returns:
        Tuple of (cropped RGB image, cropped grayscale image)
    """
    # First, try to deskew the image
    image_rgb, gray_image = deskew_image(image_rgb, gray_image)
    
    # Create a copy for visualization
    vis_image = image_rgb.copy()
    
    # Resize if the image is too large (for faster processing)
    height, width = gray_image.shape
    max_dimension = 1500
    scale = 1.0
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray_resized = cv2.resize(gray_image, (new_width, new_height))
    else:
        gray_resized = gray_image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_resized, (9, 9), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 4
    )
    
    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # If no contours found, return original images
    if not contours:
        print("No paper/document detected. Using original image.")
        return image_rgb, gray_image
    
    # Find the largest contour (assumed to be the paper)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Scale back to original size
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)
    
    # Add a small margin
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(width - x, w + 2 * margin)
    h = min(height - y, h + 2 * margin)
    
    # Draw the rectangle on the visualization image
    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Save debug visualization if path is provided
    if debug_path and image_name:
        plt.figure(figsize=(12, 10))
        plt.imshow(vis_image)
        plt.title("Detected Paper/Document")
        plt.axis('off')
        paper_vis_path = os.path.join(debug_path, f"{image_name}_paper_detection.png")
        plt.savefig(paper_vis_path)
        plt.close()
    
    # Crop the images
    cropped_rgb = image_rgb[y:y+h, x:x+w]
    cropped_gray = gray_image[y:y+h, x:x+w]
    
    return cropped_rgb, cropped_gray

def preprocess_image(gray_image, debug=False):
    """
    Preprocess the grayscale image to enhance letter detection.
    
    Args:
        gray_image: Grayscale image
        debug: Whether to return debug images
        
    Returns:
        Binary image with letters highlighted
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply adaptive thresholding to handle different lighting conditions
    # This creates a binary image where letters are white on black background
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply morphological operations to clean up the binary image
    # This helps remove small noise and fill small gaps in letters
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Apply dilation to make letters more prominent
    dilated = cv2.dilate(morph, kernel, iterations=1)
    
    if debug:
        return dilated, [blurred, binary, morph, dilated]
    return dilated

def find_contours(binary_image, min_area=500, max_area=15000):
    """
    Find contours of potential letters in the binary image.
    
    Args:
        binary_image: Binary image with letters highlighted
        min_area: Minimum contour area to consider (filters out noise)
        max_area: Maximum contour area to consider (filters out large artifacts)
        
    Returns:
        List of contours that likely represent letters
    """
    # Find all contours in the binary image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter contours by area to remove noise and non-letter objects
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out contours that are too wide or too tall (likely not letters)
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 2.0:  # Reasonable aspect ratio for letters
                filtered_contours.append(contour)
    
    # Sort contours from top-left to bottom-right (reading order)
    # This helps organize the letters in a natural sequence
    def sort_key(contour):
        x, y, w, h = cv2.boundingRect(contour)
        # Create a grid-based key for sorting (row-major order)
        row = y // 50  # Approximate row height
        return row * 1000 + x  # Row-major sorting
    
    filtered_contours.sort(key=sort_key)
    
    return filtered_contours

def extract_letter_images(original_image, gray_image, contours, target_size=(28, 28), padding=10):
    """
    Extract individual letter images from the contours.
    
    Args:
        original_image: Original RGB image
        gray_image: Grayscale image
        contours: List of contours representing letters
        target_size: Size to resize extracted letters to (matches MNIST format)
        padding: Padding to add around each letter
        
    Returns:
        List of extracted and processed letter images
    """
    letter_images = []
    bounding_boxes = []
    
    for i, contour in enumerate(contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding around the letter
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(gray_image.shape[1] - x_pad, w + 2 * padding)
        h_pad = min(gray_image.shape[0] - y_pad, h + 2 * padding)
        
        # Extract the letter region from the grayscale image
        letter_roi = gray_image[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        
        # Apply thresholding to convert to binary (black and white)
        _, letter_binary = cv2.threshold(
            letter_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Resize to target size (28x28 for MNIST compatibility)
        letter_resized = cv2.resize(letter_binary, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1] range
        letter_normalized = letter_resized.astype('float32') / 255.0
        
        letter_images.append(letter_normalized)
        bounding_boxes.append((x_pad, y_pad, w_pad, h_pad))
    
    return letter_images, bounding_boxes

def visualize_letter_detection(original_image, contours, bounding_boxes, save_path=None):
    """
    Visualize the detected letters on the original image.
    
    Args:
        original_image: Original RGB image
        contours: List of contours representing letters
        bounding_boxes: List of bounding boxes (x, y, w, h)
        save_path: Path to save the visualization image
    """
    # Create a copy of the original image for visualization
    vis_image = original_image.copy()
    
    # Draw contours and bounding boxes
    for i, (contour, bbox) in enumerate(zip(contours, bounding_boxes)):
        x, y, w, h = bbox
        
        # Draw contour
        cv2.drawContours(vis_image, [contour], 0, (0, 255, 0), 2)
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Add label (letter index)
        cv2.putText(
            vis_image, f"#{i+1}", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )
    
    # Display the image
    plt.figure(figsize=(12, 10))
    plt.imshow(vis_image)
    plt.title("Detected Letters")
    plt.axis('off')
    
    # Save the visualization if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def visualize_extracted_letters(letter_images, save_path=None):
    """
    Visualize the extracted and processed letter images.
    
    Args:
        letter_images: List of processed letter images
        save_path: Path to save the visualization image
    """
    num_letters = len(letter_images)
    rows = int(np.ceil(num_letters / 5))
    
    plt.figure(figsize=(15, 3 * rows))
    
    for i, img in enumerate(letter_images):
        plt.subplot(rows, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Letter #{i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Extracted letters visualization saved to {save_path}")
    
    plt.close()

def save_letter_images(letter_images, letter_label, letter_dirs, start_idx=0):
    """
    Save the extracted letter images to the appropriate directory.
    
    Args:
        letter_images: List of processed letter images
        letter_label: Letter label (A-E)
        letter_dirs: Dictionary of letter directories
        start_idx: Starting index for file naming
        
    Returns:
        Number of images saved
    """
    if letter_label not in letter_dirs:
        raise ValueError(f"Invalid letter label: {letter_label}")
    
    save_dir = letter_dirs[letter_label]
    
    for i, img in enumerate(letter_images):
        # Convert back to 0-255 range for saving
        img_to_save = (img * 255).astype(np.uint8)
        
        # Create filename with proper indexing
        filename = f"{letter_label}_{start_idx + i:03d}.png"
        filepath = os.path.join(save_dir, filename)
        
        # Save the image
        cv2.imwrite(filepath, img_to_save)
    
    print(f"Saved {len(letter_images)} images of letter {letter_label} to {save_dir}")
    return len(letter_images)

def process_image(image_path, letter_label, letter_dirs, debug_path, min_area=500, max_area=15000):
    """
    Process a single image containing multiple instances of a letter.
    
    Args:
        image_path: Path to the image file
        letter_label: Letter label (A-E)
        letter_dirs: Dictionary of letter directories
        debug_path: Path to save debug visualizations
        min_area: Minimum contour area for letter detection
        max_area: Maximum contour area for letter detection
        
    Returns:
        Number of letters extracted and saved
    """
    # Get image filename without extension for debug images
    image_name = Path(image_path).stem
    
    # Load the image
    original_image, gray_image = load_image(image_path)
    
    # First detect and crop to the paper/document
    print("Detecting and cropping to paper/document...")
    cropped_rgb, cropped_gray = detect_and_crop_paper(
        original_image, gray_image, debug_path, image_name
    )
    
    # Preprocess the cropped image
    binary_image, debug_images = preprocess_image(cropped_gray, debug=True)
    
    # Find contours of potential letters
    contours = find_contours(binary_image, min_area, max_area)
    
    # Extract individual letter images
    letter_images, bounding_boxes = extract_letter_images(cropped_rgb, cropped_gray, contours)
    
    # Visualize letter detection and save
    detection_vis_path = os.path.join(debug_path, f"{image_name}_detection.png")
    visualize_letter_detection(cropped_rgb, contours, bounding_boxes, detection_vis_path)
    
    # Visualize extracted letters and save
    letters_vis_path = os.path.join(debug_path, f"{image_name}_extracted.png")
    visualize_extracted_letters(letter_images, letters_vis_path)
    
    # Save the preprocessing steps for debugging
    plt.figure(figsize=(15, 5))
    titles = ["Blurred", "Binary", "Morphology", "Dilated"]
    for i, img in enumerate(debug_images):
        plt.subplot(1, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(debug_path, f"{image_name}_preprocessing.png"))
    plt.close()
    
    # Count existing images to determine starting index
    existing_count = len(os.listdir(letter_dirs[letter_label]))
    
    # Save the extracted letter images
    num_saved = save_letter_images(letter_images, letter_label, letter_dirs, existing_count)
    
    return num_saved

def process_all_images(raw_images_dir, letter_mapping=None):
    """
    Process all images in the raw images directory.
    
    Args:
        raw_images_dir: Directory containing raw images
        letter_mapping: Dictionary mapping image filenames to letter labels
        
    Returns:
        Total number of letters extracted
    """
    # Create directories for processed images
    letter_dirs, debug_path = create_directories()
    
    # Get all image files in the raw images directory
    image_files = [f for f in os.listdir(raw_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not letter_mapping:
        # If no mapping provided, prompt user for each image
        letter_mapping = {}
        for img_file in image_files:
            print(f"\nProcessing: {img_file}")
            letter = input(f"Which letter is in this image (A-E)? ").upper()
            while letter not in "ABCDE":
                letter = input("Invalid input. Please enter A, B, C, D, or E: ").upper()
            letter_mapping[img_file] = letter
    
    # Process each image
    total_letters = 0
    for img_file in image_files:
        if img_file in letter_mapping:
            letter = letter_mapping[img_file]
            print(f"\nProcessing {img_file} as letter {letter}...")
            img_path = os.path.join(raw_images_dir, img_file)
            
            # Adjust area thresholds based on the image
            # These are now relative to the cropped paper area
            min_area = 300
            max_area = 8000
                
            num_letters = process_image(img_path, letter, letter_dirs, debug_path, min_area, max_area)
            total_letters += num_letters
    
    print(f"\nTotal letters extracted and saved: {total_letters}")
    return total_letters

def main():
    """Main function to run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess handwritten letter images")
    parser.add_argument("--raw_dir", default="data/raw_images", help="Directory containing raw images")
    args = parser.parse_args()
    
    # Get the set name from the raw_dir path
    set_name = os.path.basename(args.raw_dir)
    
    # Define letter mapping based on your image filenames
    # This maps each image to the letter it contains
    if set_name == "set3":
        letter_mapping = {
            "IMG_5558.JPG": "A",
            "IMG_5559.JPG": "A",
            "IMG_5560.JPG": "A",
            "IMG_5561.JPG": "B",
            "IMG_5562.JPG": "B",
            "IMG_5563.JPG": "B",
            "IMG_5564.JPG": "C",
            "IMG_5565.JPG": "C",
            "IMG_5566.JPG": "C",
            "IMG_5567.JPG": "D",
            "IMG_5568.JPG": "D",
            "IMG_5569.JPG": "D",
            "IMG_5570.JPG": "E",
            "IMG_5571.JPG": "E",
            "IMG_5572.JPG": "E"
        }
    else:
        letter_mapping = {
            "IMG_5535.JPG": "A",
            "IMG_5536.JPG": "A",
            "IMG_5537.JPG": "A",
            "IMG_5538.JPG": "B",
            "IMG_5539.JPG": "B",
            "IMG_5540.JPG": "B",
            "IMG_5541.JPG": "C",
            "IMG_5542.JPG": "C",
            "IMG_5543.JPG": "C",
            "IMG_5544.JPG": "D",
            "IMG_5545.JPG": "D",
            "IMG_5546.JPG": "D",
            "IMG_5547.JPG": "E",
            "IMG_5548.JPG": "E",
            "IMG_5549.JPG": "E"
        }
    
    # Process all images
    process_all_images(args.raw_dir, letter_mapping)

if __name__ == "__main__":
    main() 