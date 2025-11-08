from google.cloud import storage
import os
import uuid
from datetime import timedelta

# ---------- CONFIGURATION ----------
# Path to your service account key is removed.
# On Cloud Run, authentication is handled automatically.
# For local development, use `gcloud auth application-default login`.

# Your bucket name
BUCKET_NAME = "hackcbs_generate_uri"  # change this
# ----------------------------------


def upload_to_gcs(file_path: str):
    """
    Uploads a file to Google Cloud Storage and returns a temporary signed URL.

    Args:
        file_path (str): Local path of the file to upload.

    Returns:
        str: The signed URL of the uploaded file, valid for 1 hour.
    """
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Generate a unique file name
        file_ext = file_path.split(".")[-1]
        blob_name = f"{uuid.uuid4()}.{file_ext}"

        blob = bucket.blob(blob_name)

        # Upload file
        blob.upload_from_filename(file_path)

        # Generate a temporary signed URL (1 hour validity)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET"
        )
        return url

    except Exception as e:
        return f"Error uploading file: {e}"


# ---------- Example Usage ----------
if __name__ == "__main__":
    local_path = input("Enter the path to your file: ").strip()

    if os.path.exists(local_path):
        # Example: Return a signed URL
        uri = upload_to_gcs(local_path)
        print("File uploaded successfully!")
        print("Signed URL (valid for 1 hour):", uri)
    else:
        print("File not found.")
