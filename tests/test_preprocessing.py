import unittest
import cv2
import numpy as np
import os
import sys

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessing import preprocess_image  # Correct import

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_image(self):
        # Create a dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        processed_image = preprocess_image(image)
        self.assertIsNotNone(processed_image)

if __name__ == "__main__":
    unittest.main()