from google.cloud import storage
import os
import uuid
from datetime import timedelta

# ---------- CONFIGURATION ----------
# Path to your service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "genai-hackathon-476613-85268833f6b1.json"

# Your bucket name
BUCKET_NAME = "hackcbs_generate_uri"  # change this
# ----------------------------------


def upload_to_gcs(file_path: str, make_public: bool = True, signed_url: bool = False):
    """
    Uploads a file to Google Cloud Storage and returns its URI.

    Args:
        file_path (str): Local path of the image to upload.
        make_public (bool): If True, makes the file publicly accessible.
        signed_url (bool): If True, generates a temporary signed URL.

    Returns:
        str: The public or signed URL of the uploaded file.
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

        if signed_url:
            # Generate a temporary signed URL (1 hour validity)
            url = blob.generate_signed_url(expiration=timedelta(hours=1))
            return url

        elif make_public:
            blob.make_public()
            return blob.public_url

        else:
            # Return the gs:// URI
            return f"gs://{BUCKET_NAME}/{blob_name}"

    except Exception as e:
        return f"Error uploading file: {e}"


# ---------- Example Usage ----------
if __name__ == "__main__":
    local_path = input("Enter the path to your image: ").strip()

    # Example: Return a public URI
    uri = upload_to_gcs(local_path, make_public=True)
    print("File uploaded successfully!")
    print("URI:", uri)
