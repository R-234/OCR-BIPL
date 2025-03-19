import os
import shutil
from zipfile import ZipFile
import logging
from config import IMAGE_FOLDER, OUTPUT_FOLDER, FAILED_FOLDER  # Import variables from config.py

# Configure logging
logging.basicConfig(level=logging.INFO)

def create_zip(folder, zip_name):
    """Create a ZIP file of all images in a folder."""
    try:
        zip_path = zip_name
        with ZipFile(zip_path, 'w') as zipf:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                zipf.write(file_path, arcname=file)
        return zip_path
    except Exception as e:
        logging.error(f"Error creating ZIP file: {e}")
        return None

def cleanup_folders():
    """Delete processed and failed images after zipping."""
    try:
        for folder in [IMAGE_FOLDER, OUTPUT_FOLDER, FAILED_FOLDER]:  # Use imported variables
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))
    except Exception as e:
        logging.error(f"Error cleaning up folders: {e}")