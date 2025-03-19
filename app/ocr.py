import pytesseract
import re
import logging
import os
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_text(image):
    """Extract numbers using dynamic ROI based on text detection."""
    try:
        # Use pytesseract to detect text regions
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        for i in range(len(data['text'])):
            if data['text'][i].strip():  # If text is detected
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                roi = image[y:y+h, x:x+w]
                
                # Save ROI for debugging
                debug_path = os.path.join("data", f"debug_roi_{i}.jpg")
                cv2.imwrite(debug_path, roi)
                
                extracted_text = pytesseract.image_to_string(roi, config='--psm 6 --oem 1 -l eng').strip()
                numbers_only = re.sub(r'\D', '', extracted_text)
                
                logging.info(f"Extracted text: {extracted_text}")
                logging.info(f"Numbers only: {numbers_only}")
                
                if numbers_only and len(numbers_only) >= 4:  # Allow numbers with 4 or more digits
                    return numbers_only.zfill(4)
        return None
    except Exception as e:
        logging.error(f"Error in text extraction: {e}")
        return None