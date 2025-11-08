# 1. Base Image
# Start from a lightweight, official Python image
FROM python:3.11-slim

# 2. Set Working Directory
# Set the default directory for all subsequent commands
WORKDIR /app

# 3. Copy requirements first to leverage Docker's build cache
COPY requirements.txt .

# 4. Install Python Dependencies
# --no-cache-dir saves space
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy All Project Code
# This copies all your .py files and your .json key
COPY . .

# 6. Set Environment Variable for GCS
# This tells your app where to find the GCS key inside the container.
# Make sure this filename exactly matches your key file.
ENV GOOGLE_APPLICATION_CREDENTIALS=genai-hackathon-476613-85268833f6b1.json

# 7. Expose the Port
# Tell Docker that the container will listen on port 8000
EXPOSE 8000:8080

# 8. Define the Start Command
# Run the API using uvicorn.
# --host 0.0.0.0 is CRITICAL to make it accessible from outside the container.
# We remove --reload, as that is for development only.
CMD ["uvicorn", "api:app_api", "--host", "0.0.0.0", "--port", "8000"]