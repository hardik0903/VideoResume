# Use Python 3.10 slim image to keep size down
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# ffmpeg -> for moviepy
# libgl1-mesa-glx -> for opencv
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to cache installation
COPY requirements.txt .

# Install python dependencies
# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 7860 (Hugging Face default) or 8000
EXPOSE 7860

# Start the application
# Note: Hugging Face expects access on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]