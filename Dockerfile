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
# This copies all your .py files. The .dockerignore file prevents copying secrets.
COPY . .

# 6. Expose the Port
# Tell Docker that the container will listen on port 8000.
# Cloud Run will automatically detect this port.
EXPOSE 8000

# 7. Define the Start Command
# Run the API using uvicorn.
# --host 0.0.0.0 is CRITICAL to make it accessible from outside the container.
# This command now uses the $PORT environment variable provided by Cloud Run.
CMD exec uvicorn api:app_api --host 0.0.0.0 --port ${PORT:-8000}
