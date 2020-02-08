#!/usr/bin/env bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
gdown https://drive.google.com/uc?id=1LOZYZ2I9Piw3vvYbDVR5yoXU4wI_Ns-6  # downloads multi_news.json.gz
gunzip multi_news.json.gz
