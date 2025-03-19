import streamlit as st
import cv2
import logging
import os
import shutil
import concurrent.futures
from config import IMAGE_FOLDER, OUTPUT_FOLDER, FAILED_FOLDER, ZIP_SUCCESS, ZIP_FAILED
from preprocessing import preprocess_image
from ocr import extract_text
from utils import create_zip, cleanup_folders

# Ensure folders exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FAILED_FOLDER, exist_ok=True)

def process_single_image(image_file):
    """Process a single image and return the result."""
    try:
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_file}")
        
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise ValueError("Image preprocessing failed")
        
        extracted_number = extract_text(processed_image)
        if not extracted_number:
            raise ValueError("Text extraction failed")
        
        new_filename = f"{extracted_number}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, new_filename)
        success = cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            raise ValueError("Failed to save image")
        
        return None, new_filename
    except Exception as e:
        logging.error(f"Error processing {image_file}: {e}")
        return image_file, None

def process_images_parallel(image_files):
    """Process multiple images using multi-threading."""
    renamed_files, failed_files = [], []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # Limit threads
        results = executor.map(process_single_image, image_files)

    for failed, renamed in results:
        if failed:
            failed_files.append(failed)
            src_path = os.path.join(IMAGE_FOLDER, failed)
            dst_path = os.path.join(FAILED_FOLDER, failed)
            
            # Check if the source file exists before moving
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
            else:
                logging.error(f"File not found: {src_path}")
        if renamed:
            renamed_files.append(renamed)

    return renamed_files, failed_files

def process_and_zip(uploaded_files):
    """Processes images using OCR and returns ZIP file paths."""
    image_files = []
    for file in uploaded_files:
        if file.size > 5 * 1024 * 1024:  # 5MB limit
            st.warning(f"Skipping {file.name} (too large).")
            continue
        
        file_path = os.path.join(IMAGE_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        image_files.append(file.name)

    renamed_files, failed_files = process_images_parallel(image_files)  # Use parallel processing

    success_zip = create_zip(OUTPUT_FOLDER, ZIP_SUCCESS) if renamed_files else None
    failed_zip = create_zip(FAILED_FOLDER, ZIP_FAILED) if failed_files else None

    cleanup_folders()
    return success_zip, failed_zip

# Streamlit UI
st.markdown(
    """
    <style>
    @keyframes scrollText {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    .scrolling-text {
        white-space: nowrap;
        overflow: hidden;
        position: relative;
        width: 100%;
        font-size: 20px;
        font-weight: bold;
        color: #0066cc;
    }
    .scrolling-text span {
        display: inline-block;
        animation: scrollText 10s linear infinite;
    }
    .title {
        font-size: 30px;
        font-weight: bold;
        color: #ff5733;
        text-align: center;
    }
    </style>
    <div class='scrolling-text'><span>Blacklead Infratech PVT LTD</span></div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>OCR Model for BIPL</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Process Images"):
    if uploaded_files:
        with st.spinner("Processing images..."):
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)
            success_zip, failed_zip = process_and_zip(uploaded_files)
            progress_bar.progress(1.0)  # Complete progress bar

        st.success("Processing Completed!")

        if success_zip:
            with open(success_zip, "rb") as file:
                st.download_button("Download Processed Images (ZIP)", file, ZIP_SUCCESS, "application/zip")
        if failed_zip:
            with open(failed_zip, "rb") as file:
                st.download_button("Download Failed Images (ZIP)", file, ZIP_FAILED, "application/zip")
    else:
        st.warning("Please upload images before processing.")