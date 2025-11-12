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
true_folder = 'C:/Users/Microa/OneDrive/Desktop/project/rfactual'
false_folder = 'C:/Users/Micro/OneDrive/Desktop/project/arfake'
model = None
vectorizer = None
accuracy = None
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text
def load_folder(folder_path, label):
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
    global model, vectorizer, accuracy
    print(" Loading data...")
    true_data = load_folder(true_folder, 1)
    false_data = load_folder(false_folder, 0)
    data = pd.concat([true_data, false_data], ignore_index=True)
    if len(data) > 10000:
        data = data.sample(n=10000, random_state=42)
        print(f"⚡ Sampled to {len(data)} rows for faster training")
    print("🧹 Preprocessing text...")
    data['text'] = data['text'].apply(preprocess_text)
    X = data['text']
    y = data['label']
    print(" Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2), 
        max_features=5000,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(" Training model...")
    model = LogisticRegression(
        max_iter=500,  
        C=1.0,
        solver='saga', 
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model trained successfully!")
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
def predict_news(news_text):
    """Predict if news is real or fake"""
    if model is None or vectorizer is None:
        return {"prediction": "Error", "confidence": 0, "error": "Model not trained"}
    try:
        processed_text = preprocess_text(news_text)
        vec = vectorizer.transform([processed_text])
        prediction = model.predict(vec)
        probability = model.predict_proba(vec)[0]
        
        return {
            'prediction': "Real" if prediction[0] == 1 else "Fake",
            'confidence': float(max(probability)) * 100
        }
    except Exception as e:
        return {"prediction": "Error", "confidence": 0, "error": str(e)}
