"""
Utility functions for downloading images from GCP Cloud Storage.
"""

from google.cloud import storage
import os


def download_image_from_gcs(bucket_name: str, blob_name: str, destination_folder: str = "images") -> str:
    """
    Download an image from GCP Cloud Storage bucket and save it locally.
    
    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Path to the blob (file) within the bucket
        destination_folder: Local folder to save the image (default: "images")
    
    Returns:
        Path to the downloaded file
    """
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Initialize the GCS client
    client = storage.Client()
    
    # Get the bucket and blob
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Extract filename from blob name
    filename = os.path.basename(blob_name)
    destination_path = os.path.join(destination_folder, filename)
    
    # Download the file
    blob.download_to_filename(destination_path)
    print(f"Downloaded {blob_name} to {destination_path}")
    
    return destination_path


def download_from_gcs_url(gcs_url: str, destination_folder: str = "images") -> str:
    """
    Download an image from a GCP Cloud Storage URL.
    
    Args:
        gcs_url: Full GCS URL (e.g., https://storage.cloud.google.com/bucket-name/path/to/file.png)
        destination_folder: Local folder to save the image (default: "images")
    
    Returns:
        Path to the downloaded file
    """
    # Parse the URL to extract bucket name and blob name
    # URL format: https://storage.cloud.google.com/BUCKET_NAME/BLOB_PATH?authuser=X
    from urllib.parse import urlparse, unquote
    
    parsed = urlparse(gcs_url)
    path_parts = parsed.path.lstrip('/').split('/', 1)
    
    bucket_name = path_parts[0]
    blob_name = unquote(path_parts[1]) if len(path_parts) > 1 else ""
    
    return download_image_from_gcs(bucket_name, blob_name, destination_folder)


if __name__ == "__main__":
    # Download the specific image from the GCS URL
    gcs_url = "https://storage.cloud.google.com/gcp-wow-wiq-014-test-paa-input/images/1.png?authuser=5"
    
    # Get the directory of this script and set images folder relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_folder = os.path.join(script_dir, "images")
    
    downloaded_path = download_from_gcs_url(gcs_url, images_folder)
    print(f"Image saved to: {downloaded_path}")
