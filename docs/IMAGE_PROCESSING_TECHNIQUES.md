# Image Processing Techniques for Handwritten Letter Extraction

This document summarizes the main image processing techniques used in this project and provides a primer on essential OpenCV concepts for beginners.

---

## **Techniques Used in This Project**

### 1. Grayscale Conversion
- Converts color images to grayscale for simpler processing.
- **OpenCV:** `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`

### 2. Contrast Enhancement
- Improves the distinction between letters and background.
- **Techniques:** Histogram Equalization, CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **OpenCV:** `cv2.equalizeHist`, `cv2.createCLAHE()`

### 3. Thresholding
- Converts grayscale images to binary (black and white) images.
- **Otsu’s Thresholding:** Finds a global threshold automatically.
- **Adaptive Thresholding:** Computes a local threshold for each region, useful for uneven lighting.
- **OpenCV:** `cv2.threshold`, `cv2.adaptiveThreshold`

### 4. Morphological Operations
- Modify shapes in binary images to remove noise or fill gaps.
- **Erosion:** Removes small white noise, thins objects.
- **Dilation:** Fills small holes, thickens objects.
- **Opening (Erode then Dilate):** Removes small white noise.
- **Closing (Dilate then Erode):** Fills small black holes/gaps inside letters.
- **OpenCV:** `cv2.erode`, `cv2.dilate`, `cv2.morphologyEx`

### 5. Contour Detection
- Finds the boundaries of connected components (letters, blobs).
- Can be filtered by area, aspect ratio, or shape.
- **OpenCV:** `cv2.findContours`

### 6. Perspective Transform
- “Unwarps” the paper to a top-down view, removing background and skew.
- **OpenCV:** `cv2.getPerspectiveTransform`, `cv2.warpPerspective`

### 7. Noise Reduction
- **Median Blurring:** Removes salt-and-pepper noise while preserving edges.
- **Largest Contour Filtering:** Keeps only the main letter, discards small blobs.
- **OpenCV:** `cv2.medianBlur`

### 8. Manual Adjustment (GUI)
- Allows human correction of bounding boxes, rotation, and labeling.

### 9. Saving and Indexing
- Crops, resizes, and saves each letter in a standard format (e.g., 28x28, white-on-black).
- Generates a CSV index for downstream ML tasks.

---

## **OpenCV Starter Topics: A Primer**

If you’re new to OpenCV and image processing, here are the foundational concepts and functions to learn:

### **Basic Image I/O and Display**
- Reading and writing images: `cv2.imread`, `cv2.imwrite`
- Displaying images: `cv2.imshow`, `cv2.waitKey`, `cv2.destroyAllWindows`

### **Image Types and Color Spaces**
- Grayscale vs. color images
- Color space conversion: `cv2.cvtColor`

### **Image Preprocessing**
- Resizing: `cv2.resize`
- Blurring: `cv2.GaussianBlur`, `cv2.medianBlur`
- Thresholding: `cv2.threshold`, `cv2.adaptiveThreshold`
- Contrast enhancement: `cv2.equalizeHist`, `cv2.createCLAHE`

### **Morphological Operations**
- Erosion and dilation: `cv2.erode`, `cv2.dilate`
- Opening and closing: `cv2.morphologyEx`
- Structuring elements (kernels): `cv2.getStructuringElement`, `np.ones((k, k), np.uint8)`

### **Geometric Transformations**
- Rotation, translation, scaling: `cv2.getRotationMatrix2D`, `cv2.warpAffine`
- Perspective transform: `cv2.getPerspectiveTransform`, `cv2.warpPerspective`

### **Contour and Shape Analysis**
- Finding contours: `cv2.findContours`
- Drawing contours: `cv2.drawContours`
- Bounding rectangles: `cv2.boundingRect`
- Filtering by area, aspect ratio

### **Feature Extraction and Segmentation**
- Edge detection: `cv2.Canny`
- Connected components: `cv2.connectedComponents`
- Skeletonization (advanced)

### **User Interaction**
- Mouse callbacks for drawing/selecting: `cv2.setMouseCallback`
- Keyboard input: `cv2.waitKey`

### **Saving Results**
- Writing images: `cv2.imwrite`
- Exporting data: CSV, JSON, etc.

---

## **Further Learning**
- [OpenCV-Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [PyImageSearch Blog](https://pyimagesearch.com/)
- [Scikit-Image Documentation](https://scikit-image.org/)
- [Digital Image Processing Textbooks](https://www.imageprocessingplace.com/)

---

This primer and summary should help you get started with OpenCV and understand the techniques used in this project. For more details, see the code and comments in the provided scripts! 