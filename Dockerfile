# Use Python 3.9 or 3.10 slim image
FROM python:3.10-slim

# Set the working directory to /code
WORKDIR /code

# Install system dependencies required for MoviePy and OpenCV
# ffmpeg is needed for audio extraction
# libgl1-mesa-glx is needed for cv2
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application files
COPY . .

# Create a directory for cache/temp files and give permissions
# (FastAPI/MoviePy needs to write temp files)
RUN chmod -R 777 /code

# Expose port 7860 (Required by Hugging Face Spaces)
EXPOSE 7860

# Start the application on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]