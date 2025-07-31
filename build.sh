#!/bin/bash
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
echo "✅ Build script completed successfully."
