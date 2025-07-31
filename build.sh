#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Then download required NLP models
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
