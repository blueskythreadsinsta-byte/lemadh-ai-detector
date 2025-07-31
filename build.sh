#!/bin/bash

# Upgrade pip and install packages without cache
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

# Download required NLP models
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm

echo "✅ Build script completed successfully."
