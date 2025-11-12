# News Detector (news_det.py)

Small Flask-based prototype that trains a simple text classifier to detect news categories (or binary news detection) using TF-IDF features and logistic regression. The main script is `news_det.py`.
#THE ALL DATA IN DATA SETS ARE FROM KAGGLE 
## What this file does
- Loads and preprocesses text data (CSV/TSV or pandas-readable formats).
- Converts raw text into TF-IDF features.
- Trains a Logistic Regression classifier.
- Exposes a minimal Flask API to make predictions from HTTP requests.
- Evaluates model performance with common metrics.

## Key imports and why they are used
- `os` — file and environment operations (paths, existence checks).
- `pandas as pd` — load and manipulate datasets (CSV, DataFrame operations).
- `numpy as np` — numeric operations and array handling used by sklearn/pandas.
- `sklearn.model_selection.train_test_split` — split dataset for train/test.
- `sklearn.feature_extraction.text.TfidfVectorizer` — convert text to TF-IDF feature vectors.
- `sklearn.linear_model.LogisticRegression` — lightweight, fast linear classifier for text.
- `sklearn.metrics.accuracy_score, classification_report` — evaluate model accuracy and per-class metrics.
- `flask (Flask, render_template_string, request, jsonify)` — lightweight web server and JSON endpoints for model inference.
- `re` — regular expressions for basic text cleaning and normalization.

## Quick start
1. Create a virtual environment (recommended):
   - Windows (PowerShell):
     - python -m venv .venv
     - .\.venv\Scripts\Activate.ps1
2. Install dependencies:
   - pip install pandas numpy scikit-learn flask
3. Run the script:
   - python news_det.py
4. Use the HTTP endpoint exposed by the Flask app to send text for prediction (example endpoint is described in the script).

## Notes
- The implementation assumes you will provide a properly formatted dataset and adjust preprocessing as needed.
- For production use, add input validation, persistent model saving/loading, and better error handling.

    #NOTICE
    FOR THIS PROGRAM YOU HAVE TO MAKE A SPECIFIC FOLDER TO PUTTING DATA.CSV FILES IT USES IT AND TRAINS WITHIN YOUR COMPUTER SO MAKE SURE YOU MAKE A rfactual and arfake  folder insde your program file to run it
