#!/bin/bash

# This script creates the necessary directories and initializes the file parameters/max_page.txt with the value "100".

# 0. Create the .env file for the AirFLOW_UID
echo "Creating the .env file..."
echo "AIRFLOW_UID=50000" > .env

# 1. Create the directories data/raw and data/cleaned
echo "Creating directories data/raw and data/cleaned..."
mkdir -p data/raw
mkdir -p data/cleaned

# Check if the directories were created successfully
if [ -d "data/raw" ] && [ -d "data/cleaned" ]; then
    echo "Directories data/raw and data/cleaned were created successfully."
else
    echo "Error: Failed to create directories."
    exit 1
fi

# 2. Create the file parameters/max_page.txt and write the value "50"
echo "Creating the file parameters/max_page.txt..."
mkdir -p parameters
echo "50" > parameters/max_page.txt

# Check if the file was created and contains the expected value
if [ -f "parameters/max_page.txt" ]; then
    echo "File parameters/max_page.txt was created successfully."
    echo "File content: $(cat parameters/max_page.txt)"
else
    echo "Error: Failed to create the file parameters/max_page.txt."
    exit 1
fi

echo "Script completed successfully."