#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Letter Image Processor

This script processes images of handwritten letters to:
1. Detect and isolate the paper from background
2. Find and draw bounding boxes around individual letters
3. Allow manual adjustment of bounding boxes
4. Export letter images in MNIST-like format (28x28, white on black)

Usage:
    python letters_processor.py [image_path]
"""

import cv2
import numpy as np
import os
import csv
import argparse
from datetime import datetime
import sys
import shutil

class LetterProcessor:
    """
    Main class for processing letter images and managing the interactive UI
    """
    
    def __init__(self, image_path, output_dir='data/processed_letters'):
        """
        Initialize the letter processor
        
        Args:
            image_path (str): Path to the input image
            output_dir (str): Directory to save processed images
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.paper_contour = None
        self.paper_mask = None
        self.bounding_boxes = []
        self.current_box = None
        self.dragging = False
        self.resizing = False
        self.selected_box_idx = -1
        self.resize_handle = None
        self.rotation_angle = 0
        self.letter_label = None
        self.window_name = "Letter Processor"
        self.scale_factor = 1.0  # For scaling between display and original image
        
        # Constants
        self.MNIST_SIZE = 28  # MNIST images are 28x28 pixels
    
    def load_image(self):
        """Load and perform initial processing of the image"""
        # Load image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_path}")
        
        # Create a copy for processing
        self.processed_image = self.original_image.copy()
        self.display_image = self.original_image.copy()
        
        # Resize if the image is too large for display
        max_display_dim = 1200
        h, w = self.original_image.shape[:2]
        if max(h, w) > max_display_dim:
            self.scale_factor = max_display_dim / max(h, w)
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            self.display_image = cv2.resize(self.display_image, (new_w, new_h))
            
        print(f"Loaded image: {self.image_path} with dimensions {self.original_image.shape}")
    
    def crop_inside_contour(self, image, contour, margin=10):
        """
        Crop just inside the bounding rectangle of the contour, with a margin.
        Args:
            image: The image to crop
            contour: The contour to use for cropping
            margin: Number of pixels to crop inside the bounding box
        Returns:
            Cropped image, and the (x, y) offset of the crop
        """
        x, y, w, h = cv2.boundingRect(contour)
        # Shrink the rectangle by 'margin' pixels on all sides
        x_new = x + margin
        y_new = y + margin
        w_new = max(1, w - 2 * margin)
        h_new = max(1, h - 2 * margin)
        cropped = image[y_new:y_new + h_new, x_new:x_new + w_new]
        return cropped, (x_new, y_new)

    def enhance_contrast(self, gray_img):
        """
        Enhance the contrast of a grayscale image using CLAHE.
        Args:
            gray_img: Grayscale image
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_img)
        return enhanced

    def get_paper_corners(self, contour):
        """
        Approximate the contour to 4 corners (for perspective transform).
        Returns a 4x2 numpy array of corner points (order: tl, tr, br, bl)
        """
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
        else:
            # Fallback: use bounding box corners
            x, y, w, h = cv2.boundingRect(contour)
            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        # Order points: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]      # top-left
        ordered[2] = pts[np.argmax(s)]      # bottom-right
        ordered[1] = pts[np.argmin(diff)]   # top-right
        ordered[3] = pts[np.argmax(diff)]   # bottom-left
        return ordered

    def warp_paper(self, image, corners):
        """
        Apply a perspective transform to get a top-down view of the paper.
        Returns the warped image.
        """
        (tl, tr, br, bl) = corners
        # Compute width and height of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def stretch_histogram(self, gray_img):
        """
        Stretch the histogram so the paper is white and letters are black.
        """
        # Use normalization to [0,255]
        norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
        # Optionally, apply histogram equalization
        eq = cv2.equalizeHist(norm)
        return eq

    def detect_paper(self):
        """
        Detect the paper, crop it using a perspective transform, and prepare for letter detection.
        Save debug images at each step.
        """
        debug_dir = os.path.join('figures', 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(debug_dir, '01_gray.png'), gray)
        # Step 2: Blur and threshold to find paper
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(debug_dir, '02_thresh.png'), thresh)
        # Step 3: Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found for paper detection.")
            return
        paper_contour = max(contours, key=cv2.contourArea)
        self.paper_contour = paper_contour
        # Step 4: Find corners and warp
        corners = self.get_paper_corners(paper_contour)
        warped = self.warp_paper(self.original_image, corners)
        cv2.imwrite(os.path.join(debug_dir, '03_warped.png'), warped)
        # Step 4.5: Crop a margin from the edge to ensure table is excluded
        EDGE_MARGIN = 25  # pixels to crop from each edge (adjust as needed)
        h, w = warped.shape[:2]
        cropped_warped = warped[EDGE_MARGIN:h-EDGE_MARGIN, EDGE_MARGIN:w-EDGE_MARGIN]
        cv2.imwrite(os.path.join(debug_dir, '03b_cropped_warped.png'), cropped_warped)
        # Step 5: Grayscale and threshold the cropped warped image for detection
        warped_gray = cv2.cvtColor(cropped_warped, cv2.COLOR_BGR2GRAY)
        _, warped_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(debug_dir, '04_warped_thresh.png'), warped_thresh)
        # Store for use in letter detection
        self.processed_image = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)  # for UI
        self.letter_detection_image = warped_thresh  # for detection
        h, w = warped_gray.shape[:2]
        new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
        self.display_image = cv2.resize(self.processed_image, (new_w, new_h))
        print("Paper detected, warped, cropped inside edge, and thresholded for letter detection. Debug images saved in figures/debug/.")

    def auto_detect_letters(self):
        """
        Automatically detect letters in the image and create initial bounding boxes
        Uses adaptive thresholding and closing only for robust detection of open shapes (like 'C')
        """
        # Use the thresholded warped image for detection
        if hasattr(self, 'letter_detection_image'):
            gray = self.letter_detection_image
        else:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        # Adaptive thresholding for detection
        thresh_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # Morphological closing only (dilate then erode)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours by size to avoid noise
        img_area = thresh_img.shape[0] * thresh_img.shape[1]
        min_area = img_area * 0.001  # 0.1% of image area
        max_area = img_area * 0.1    # 10% of image area
        BOX_MARGIN = 18  # pixels to expand each bounding box (adjust as needed)
        letter_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        # Create bounding boxes with margin
        self.bounding_boxes = []
        for contour in letter_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x = max(0, x - BOX_MARGIN)
            y = max(0, y - BOX_MARGIN)
            w = min(thresh_img.shape[1] - x, w + 2 * BOX_MARGIN)
            h = min(thresh_img.shape[0] - y, h + 2 * BOX_MARGIN)
            self.bounding_boxes.append((x, y, w, h))
        print(f"Auto-detected {len(self.bounding_boxes)} potential letters (adaptive threshold + closing only)")
        # If no letters detected or too few, try to use the grid approach
        if len(self.bounding_boxes) < 5:  # We expect 9 letters
            print("Too few letters detected. Trying grid arrangement...")
            self.auto_arrange_grid()

    def process(self):
        """Run the full processing pipeline"""
        self.load_image()
        self.detect_paper()
        self.auto_detect_letters()
        self.run_ui()

    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for the interactive UI
        Args:
            event: OpenCV mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        # Convert display coordinates to original image coordinates
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
        # Left button down
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on a box
            for i, (bx, by, bw, bh) in enumerate(self.bounding_boxes):
                # Scale box coordinates for display
                disp_x = int(bx * self.scale_factor)
                disp_y = int(by * self.scale_factor)
                disp_w = int(bw * self.scale_factor)
                disp_h = int(bh * self.scale_factor)
                # Check if clicking on resize handle
                if self.is_on_resize_handle(x, y, disp_x, disp_y, disp_w, disp_h):
                    self.selected_box_idx = i
                    self.resizing = True
                    self.resize_handle = self.get_resize_handle(x, y, disp_x, disp_y, disp_w, disp_h)
                    break
                # Check if clicking inside a box
                elif disp_x <= x <= disp_x + disp_w and disp_y <= y <= disp_y + disp_h:
                    self.selected_box_idx = i
                    self.dragging = True
                    self.drag_start_x = x
                    self.drag_start_y = y
                    break
            else:  # Not clicking on any existing box
                self.current_box = (orig_x, orig_y, 0, 0)
                self.dragging = True
                self.selected_box_idx = -1
        # Mouse move
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                if self.selected_box_idx >= 0:  # Moving existing box
                    bx, by, bw, bh = self.bounding_boxes[self.selected_box_idx]
                    # Calculate movement in original image coordinates
                    dx = int((x - self.drag_start_x) / self.scale_factor)
                    dy = int((y - self.drag_start_y) / self.scale_factor)
                    self.bounding_boxes[self.selected_box_idx] = (bx + dx, by + dy, bw, bh)
                    self.drag_start_x = x
                    self.drag_start_y = y
                else:  # Creating new box
                    self.current_box = (
                        min(self.current_box[0], orig_x),
                        min(self.current_box[1], orig_y),
                        abs(orig_x - self.current_box[0]),
                        abs(orig_y - self.current_box[1])
                    )
            elif self.resizing and self.selected_box_idx >= 0:
                bx, by, bw, bh = self.bounding_boxes[self.selected_box_idx]
                # Convert display coordinates to original image coordinates
                orig_x = int(x / self.scale_factor)
                orig_y = int(y / self.scale_factor)
                # Resize based on which handle is being dragged
                if self.resize_handle == "top-left":
                    new_w = bw + (bx - orig_x)
                    new_h = bh + (by - orig_y)
                    new_x = orig_x
                    new_y = orig_y
                elif self.resize_handle == "top-right":
                    new_w = orig_x - bx
                    new_h = bh + (by - orig_y)
                    new_x = bx
                    new_y = orig_y
                elif self.resize_handle == "bottom-left":
                    new_w = bw + (bx - orig_x)
                    new_h = orig_y - by
                    new_x = orig_x
                    new_y = by
                elif self.resize_handle == "bottom-right":
                    new_w = orig_x - bx
                    new_h = orig_y - by
                    new_x = bx
                    new_y = by
                # Update box if dimensions are valid
                if new_w > 10 and new_h > 10:
                    self.bounding_boxes[self.selected_box_idx] = (new_x, new_y, new_w, new_h)
        # Left button up
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging and self.selected_box_idx == -1 and self.current_box:
                # Finalize new box if it's big enough
                if self.current_box[2] > 10 and self.current_box[3] > 10:
                    self.bounding_boxes.append(self.current_box)
                    self.selected_box_idx = len(self.bounding_boxes) - 1
            self.dragging = False
            self.resizing = False
            self.current_box = None
        # Right button down or Control+click (for Mac)
        elif event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY):
            # Find which box was clicked
            for i, (bx, by, bw, bh) in enumerate(self.bounding_boxes):
                # Scale box coordinates for display
                disp_x = int(bx * self.scale_factor)
                disp_y = int(by * self.scale_factor)
                disp_w = int(bw * self.scale_factor)
                disp_h = int(bh * self.scale_factor)
                if disp_x <= x <= disp_x + disp_w and disp_y <= y <= disp_y + disp_h:
                    del self.bounding_boxes[i]
                    self.selected_box_idx = -1
                    print(f"Deleted box {i}")
                    break

    def is_on_resize_handle(self, x, y, bx, by, bw, bh):
        """Check if mouse is on a resize handle"""
        handle_radius = 5
        corners = [
            (bx, by),               # Top-left
            (bx + bw, by),          # Top-right
            (bx, by + bh),          # Bottom-left
            (bx + bw, by + bh)      # Bottom-right
        ]
        for corner in corners:
            if abs(corner[0] - x) < handle_radius and abs(corner[1] - y) < handle_radius:
                return True
        return False

    def get_resize_handle(self, x, y, bx, by, bw, bh):
        """Determine which resize handle the mouse is on"""
        handle_radius = 5
        # Check each corner
        if abs(bx - x) < handle_radius and abs(by - y) < handle_radius:
            return "top-left"
        elif abs(bx + bw - x) < handle_radius and abs(by - y) < handle_radius:
            return "top-right"
        elif abs(bx - x) < handle_radius and abs(by + bh - y) < handle_radius:
            return "bottom-left"
        elif abs(bx + bw - x) < handle_radius and abs(by + bh - y) < handle_radius:
            return "bottom-right"
        return None

    def run_ui(self):
        """Run the interactive UI for adjusting bounding boxes"""
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        # Instructions
        print("\n=== Letter Processor UI Controls ===")
        print("Left-click and drag: Create new box or move selected box")
        print("Right-click or Control+click (Mac): Delete selected box")
        print("'r': Rotate image 90° clockwise")
        print("'l': Rotate image 90° counter-clockwise")
        print("'t': Rotate image 180° (flip)")
        print("'g': Auto-arrange 3x3 grid")
        print("'a'-'e': Assign letter label (A-E)")
        print("'s': Save processed letters")
        print("'d': Delete selected box (alternative to right-click)")
        print("'q' or ESC: Quit\n")
        # Main UI loop
        while True:
            # Create a copy of the display image to draw on
            img_display = self.display_image.copy()
            # Draw all bounding boxes
            for i, (x, y, w, h) in enumerate(self.bounding_boxes):
                # Scale coordinates for display
                disp_x = int(x * self.scale_factor)
                disp_y = int(y * self.scale_factor)
                disp_w = int(w * self.scale_factor)
                disp_h = int(h * self.scale_factor)
                color = (0, 255, 0) if i == self.selected_box_idx else (0, 0, 255)
                cv2.rectangle(img_display, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), color, 2)
                # Draw resize handles if box is selected
                if i == self.selected_box_idx:
                    # Draw corner handles
                    handle_radius = 5
                    corners = [
                        (disp_x, disp_y),               # Top-left
                        (disp_x + disp_w, disp_y),      # Top-right
                        (disp_x, disp_y + disp_h),      # Bottom-left
                        (disp_x + disp_w, disp_y + disp_h)  # Bottom-right
                    ]
                    for corner in corners:
                        cv2.circle(img_display, corner, handle_radius, (255, 0, 0), -1)
            # Show current letter label if set
            if self.letter_label:
                cv2.putText(
                    img_display, f"Label: {self.letter_label}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
            # Display the image
            cv2.imshow(self.window_name, img_display)
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break
            elif key == ord('r'):  # Rotate clockwise
                self.rotate_image(90)
                self.detect_paper()
                self.auto_detect_letters()
            elif key == ord('l'):  # Rotate counter-clockwise
                self.rotate_image(-90)
                self.detect_paper()
                self.auto_detect_letters()
            elif key == ord('t'):  # Rotate 180°
                self.rotate_image(180)
                self.detect_paper()
                self.auto_detect_letters()
            elif key == ord('g'):  # Auto-arrange grid
                self.auto_arrange_grid()
            elif key == ord('s'):  # Save
                self.save_letters()
            elif key == ord('d') and self.selected_box_idx >= 0:  # Delete selected box
                del self.bounding_boxes[self.selected_box_idx]
                self.selected_box_idx = -1
                print("Deleted selected box")
            elif key >= ord('a') and key <= ord('e'):  # Assign letter label
                self.letter_label = chr(key).upper()
                print(f"Set letter label to: {self.letter_label}")
        # Clean up
        cv2.destroyAllWindows()

def main():
    """Main function to parse arguments and run the processor"""
    parser = argparse.ArgumentParser(description='Process images of handwritten letters')
    parser.add_argument('image_path', nargs='?', help='Path to the input image')
    parser.add_argument('--output', '-o', default='data/processed_letters',
                        help='Directory to save processed images')
    args = parser.parse_args()
    
    # If no image path provided, show a file dialog
    image_path = args.image_path
    if not image_path:
        print("Please provide an image path as an argument")
        return
    
    try:
        processor = LetterProcessor(image_path, args.output)
        processor.process()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
