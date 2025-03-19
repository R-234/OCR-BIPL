import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define data directories
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "upload_images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "renamed_images")
FAILED_FOLDER = os.path.join(BASE_DIR, "data", "failed_images")

# Define ZIP file paths
ZIP_SUCCESS = os.path.join(BASE_DIR, "data", "processed_results.zip")
ZIP_FAILED = os.path.join(BASE_DIR, "data", "failed_results.zip")

# Ensure directories exist
for folder in [IMAGE_FOLDER, OUTPUT_FOLDER, FAILED_FOLDER]:
    os.makedirs(folder, exist_ok=True)