#!/bin/bash

# Activate virtual environment if needed (Render does this internally)

# Install Python packages
pip install -r requirements.txt

# Download NLTK + SpaCy models
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm

# Confirm build succeeded
echo "✅ Build script completed successfully."
