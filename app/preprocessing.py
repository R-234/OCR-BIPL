import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def resize_image(image, max_size=1200):
    """Resize image to a maximum width/height while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image

def preprocess_image(image):
    """High-quality preprocessing: Resize, grayscale, CLAHE, noise reduction, adaptive thresholding."""
    try:
        # Resize image
        image = resize_image(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Reduce noise using Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Optional: Morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return morph
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        return None