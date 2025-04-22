# Use a python image as the basi
FROM python:3.8-slim

# Define the working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file into the container
COPY requirements.txt .

# Copy source code
COPY config.yaml .
COPY main.py .
COPY Dockerfile .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8050
EXPOSE 8050


