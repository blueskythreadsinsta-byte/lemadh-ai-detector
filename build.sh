#!/bin/bash

# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download NLP models
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
