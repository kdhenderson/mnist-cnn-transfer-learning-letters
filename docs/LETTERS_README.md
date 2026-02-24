# Letter Image Processor

This project provides scripts for processing images of handwritten letters (A-E) to create MNIST-like datasets for machine learning. The scripts detect and extract individual letters from images, with options for interactive adjustment and robust noise reduction.

---

## **Final Working Scripts**

### 1. `src/letters_processor_fixnoise.py` (**Recommended**)
- **Purpose:**  
  The most robust and recommended script for extracting all letters (A-E) from images, including those with open shapes (like “C”) and closed shapes (like “A”, “B”, “D”, “E”).
- **Features:**  
  - Detects and isolates the paper from the background.
  - Uses adaptive thresholding and morphological closing for robust letter detection.
  - Interactive GUI for manual adjustment of bounding boxes and rotation.
  - **Advanced noise reduction:**  
    - Applies multiple morphological operations and contour filtering to eliminate noise in the interior blank spaces of letters (e.g., the loop in “B”, the triangle in “A”).
  - Saves each letter as a 28x28 white-on-black PNG, with a CSV index.
- **Usage:**  
  ```bash
  python src/letters_processor_fixnoise.py data/raw_images/set3/IMG_5558.JPG
  ```
  (See UI controls below.)

### 2. `src/letters_processor.py`
- **Purpose:**  
  Similar to `letters_processor_fixnoise.py` but **does not include the advanced noise reduction**. Letters are detected and extracted, but you may see more noise inside the blank spaces of letters.
- **When to use:**  
  If you want a simpler pipeline or to compare the effect of noise reduction.

---

## **Reference and Debug Scripts**

### 3. `src/letters_processor_adaptiveC_debug.py`
- **Purpose:**  
  Experimental script focused on improving detection of open letters like “C”.
- **What it did well:**  
  - Used adaptive thresholding and closing to successfully detect “C”, which was previously missed by other methods.
- **Limitations:**  
  - Did not have a fully working GUI for manual adjustment and saving.
- **Why keep it:**  
  - Demonstrates the importance of tuning detection for open shapes and was a key step in developing the final approach.

### 4. `src/preprocess_letters_debug.py`
- **Purpose:**  
  Batch-processing script for extracting letters from images without a GUI.
- **What it did well:**  
  - Provided ideas for adaptive thresholding and morphological operations.
  - Worked well for detecting “C” and sometimes “E”.
- **Limitations:**  
  - Not robust for all letters, especially closed shapes.
  - No interactive adjustment.
- **Why keep it:**  
  - Useful for batch ideas and for understanding the strengths/weaknesses of different preprocessing pipelines.

---

## **Techniques Employed**

- **Paper Detection:**  
  - Finds the largest contour in the image and applies a perspective transform to crop out the paper, removing background and table edges.

- **Letter Detection:**  
  - Uses adaptive thresholding (Gaussian) to handle uneven lighting and both open and closed letter shapes.
  - Applies morphological closing to connect weak edges and ensure open shapes like “C” are detected.
  - Bounding boxes are expanded with a margin to avoid cropping off parts of letters.

- **Noise Reduction (in `letters_processor_fixnoise.py`):**  
  - After extracting each letter, applies:
    - Stronger erosion before opening to remove small specks.
    - Multiple morphological opens to further clean up noise.
    - Keeps only the largest contour to eliminate small blobs inside blank spaces (e.g., inside “B”, “A”).
  - This results in much cleaner letter images, especially in the interior blank spaces.

- **Interactive GUI:**  
  - Allows manual adjustment of bounding boxes, rotation, and saving of processed letters.
  - See UI controls below.

---

## **UI Controls (for interactive scripts)**

- **Left-click and drag:** Create a new bounding box
- **Left-click inside a box and drag:** Move the selected box
- **Left-click on a corner handle and drag:** Resize the selected box
- **Right-click on a box:** Delete the box
- **'r' key:** Rotate image 90° clockwise
- **'l' key:** Rotate image 90° counter-clockwise
- **'t' key:** Rotate image 180°
- **'g' key:** Auto-arrange 3x3 grid of bounding boxes
- **'a'-'e' keys:** Assign letter label (A-E)
- **'s' key:** Save processed letters
- **'q' key or ESC:** Quit

---

## **Summary**

- Use `letters_processor_fixnoise.py` for the best results on all letters, with minimal interior noise.
- Use `letters_processor.py` for a simpler pipeline (may have more noise).
- Reference/debug scripts are kept for their contributions to the final solution and for future experimentation.

--- 

## **Additional Notes**

- **Pen Type:**
  - The letters in this dataset were written with a regular Sharpie marker, which produces thick, high-contrast strokes.
  - If you use a fine-tipped Sharpie or a ball-point pen, detection may not work as well because thin strokes may be lost or have low contrast.
  - To improve detection for thin or low-contrast writing, try:
    - Reducing the amount of erosion and morphological opening.
    - Using a smaller kernel for morphological operations.
    - Increasing the image contrast (e.g., with histogram equalization or CLAHE).
    - Adjusting adaptive thresholding parameters (block size, C value).
    - Using edge detection (e.g., Canny) or skeletonization for very thin lines.
    - Experimenting with deep learning-based segmentation if classical methods fail.

- **Learn More:**
  - See `IMAGE_PROCESSING_TECHNIQUES.md` for a summary of the image processing techniques used in this project and a primer on OpenCV starter topics for beginners.

--- 