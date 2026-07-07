"""
This script automates the downloading and initial directory setup for the 
sample OpenDroneMap (ODM) 'Aukerman' drone imagery dataset.

It safely prepares the workspace for the stitching pipeline by performing 
the following sequence of operations:
  1. Verifies if the target directory (`data/input`) already exists and 
     contains files to avoid redundant network requests.
  2. Downloads the zipped repository directly from the ODM GitHub master branch.
  3. Extracts the archive into a temporary location within the data folder.
  4. Flattens the directory structure by moving all image files directly 
     into the `data/input` directory.
  5. Cleans up the environment by deleting the temporary .zip archive and 
     the extracted repository folders.
"""


import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

def setup_and_download_dataset():
    # Define dataset URL and paths
    dataset_url = "https://github.com/OpenDroneMap/odm_data_aukerman/archive/refs/heads/master.zip"
    data_dir = Path("data")
    input_dir = data_dir / "input"
    zip_path = data_dir / "dataset.zip"

    # Check if the directory already exists and contains files
    if input_dir.exists() and any(input_dir.iterdir()):
        print(f"Directory '{input_dir}' already exists and contains files")
        print("Skipping download")
        return

    # Create directories if they do not exist
    print(f"\nCreating directory '{input_dir}'...")
    input_dir.mkdir(parents=True, exist_ok=True)

    # Download the dataset from GitHub
    print("Starting direct from GitHub...")
    
    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("\nDownload complete! Extracting files...")
        
        # Extract the zip file into the base data directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Clean up by deleting the downloaded .zip file
        os.remove(zip_path)
        
        # Define the paths of the extracted GitHub folder and its internal images folder
        extracted_repo_dir = data_dir / "odm_data_aukerman-master"
        images_dir = extracted_repo_dir / "images"
        
        # Move all images from the repo folder directly into the input_dir
        print("Moving images into the input directory...")
        for image_file in images_dir.iterdir():
            if image_file.is_file():
                shutil.move(str(image_file), str(input_dir / image_file.name))
                
        # Clean up by deleting the entire extracted GitHub repository folder
        print("Cleaning up temporary repository files...")
        shutil.rmtree(extracted_repo_dir)
        
        print(f"Extraction successful! You can find your drone images at:\n👉 {input_dir}")
        
    except Exception as e:
        print(f"\nERROR: Something went wrong during download or extraction.\nDetails: {e}")

if __name__ == "__main__":
    setup_and_download_dataset()