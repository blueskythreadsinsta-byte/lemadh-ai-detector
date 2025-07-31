#!/bin/bash

# Install Python packages (force reinstall to avoid cache issues)
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Download models
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm

# Confirm build success
echo "✅ Build script completed successfully."
