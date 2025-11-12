import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template_string, request, jsonify
import re
app = Flask(__name__)
# Paths to your folders
true_folder = 'C:/Users/Micro/OneDrive/Desktop/project/rfactual'
false_folder = 'C:/Users/Micro/OneDrive/Desktop/project/arfake'
model = None
vectorizer = None
accuracy = None
def preprocess_text(text):
    """Fast text preprocessing"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text
def load_folder(folder_path, label):
    """Load CSV files - with row limit for speed"""
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(folder_path, filename), nrows=5000)
                df['label'] = label
                data_list.append(df)
            except Exception as e:
                print(f"Skipping {filename}: {e}")  
    if data_list:
        return pd.concat(data_list, ignore_index=True)
    return pd.DataFrame()
def train_model():
    """Train optimized model for speed"""
    global model, vectorizer, accuracy
    print(" Loading data...")
    true_data = load_folder(true_folder, 1)
    false_data = load_folder(false_folder, 0)
    data = pd.concat([true_data, false_data], ignore_index=True)
    