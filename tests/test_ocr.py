import unittest
import cv2
import numpy as np
import os
import sys

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ocr import extract_text  # Correct import

class TestOCR(unittest.TestCase):
    def test_extract_text(self):
        # Create a dummy image with text
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(image, "1234", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        extracted_text = extract_text(image)
        self.assertEqual(extracted_text, "1234")

if __name__ == "__main__":
    unittest.main()