import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template_string, request, jsonify
import re
import requests
app = Flask(__name__)
true_folder = 'C:/Users/Micro/OneDrive/Desktop/project/rfactual'
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
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("🔢 Vectorizing text...")
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
    print("🤖 Training model...")
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
    print(f"\n Model trained successfully!")
    print(f" Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f" Training samples: {len(X_train)}, Test samples: {len(X_test)}")
def predict_news(news_text):
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

# Ultra Modern HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detector | AI-Powered Analysis</title>
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #ffffff;
            color: #1e293b;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Clean Professional Background */
        .background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 50%, #f1f5f9 100%);
        }

        .background::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 50%);
            animation: subtleMove 30s ease infinite;
        }

        @keyframes subtleMove {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 30px;
            position: relative;
        }

        /* Professional Header */
        .header {
            text-align: center;
            margin-bottom: 60px;
            padding: 60px 0 40px;
            animation: fadeInDown 0.6s ease;
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .logo-container {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 90px;
            height: 90px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.2);
        }

        .logo {
            font-size: 3em;
        }

        .header h1 {
            font-size: 3em;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 16px;
            letter-spacing: -2px;
            line-height: 1.1;
        }

        .header .subtitle {
            font-size: 1.2em;
            color: #64748b;
            font-weight: 400;
            letter-spacing: 0.3px;
            margin-bottom: 20px;
        }

        .badge-container {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 24px;
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 50px;
            font-size: 0.85em;
            font-weight: 600;
            color: #3b82f6;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        .badge.primary {
            background: #3b82f6;
            color: #ffffff;
            border-color: #3b82f6;
        }

        /* Stats Card */
        .stats-card {
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 16px;
            padding: 40px;
            margin-bottom: 40px;
            animation: fadeIn 0.6s ease 0.1s both;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(15px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 30px;
        }

        .stat-item {
            text-align: center;
            padding: 24px;
            background: #f8fafc;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }

        .stat-item:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            border-color: #3b82f6;
        }

        .stat-value {
            font-size: 2.2em;
            font-weight: 800;
            color: #3b82f6;
            margin-bottom: 8px;
        }

        .stat-label {
            color: #64748b;
            font-size: 0.9em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        }

        /* Main Card */
        .main-card {
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 20px;
            padding: 50px;
            animation: fadeIn 0.6s ease 0.2s both;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
        }

        .card-header {
            margin-bottom: 35px;
            padding-bottom: 25px;
            border-bottom: 2px solid #f1f5f9;
        }

        .card-title {
            font-size: 1.8em;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 8px;
        }

        .card-description {
            color: #64748b;
            font-size: 0.95em;
            line-height: 1.6;
        }

        .form-group {
            margin-bottom: 30px;
        }

        label {
            display: block;
            font-weight: 600;
            margin-bottom: 12px;
            color: #1e293b;
            font-size: 1em;
            letter-spacing: 0.2px;
        }

        textarea {
            width: 100%;
            padding: 18px;
            background: #f8fafc;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            color: #1e293b;
            font-size: 15px;
            font-family: inherit;
            resize: vertical;
            transition: all 0.25s ease;
            line-height: 1.7;
            min-height: 260px;
        }

        textarea:focus {
            outline: none;
            border-color: #3b82f6;
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.08);
        }

        textarea::placeholder {
            color: #94a3b8;
        }

        .char-count {
            text-align: right;
            color: #94a3b8;
            font-size: 0.85em;
            margin-top: 10px;
            font-weight: 500;
            font-variant-numeric: tabular-nums;
        }

        .button-group {
            display: flex;
            gap: 16px;
            margin-top: 35px;
        }

        button {
            flex: 1;
            padding: 16px 40px;
            font-size: 1em;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.25s ease;
            text-transform: none;
            letter-spacing: 0.3px;
            position: relative;
        }

        .btn-analyze {
            background: #3b82f6;
            color: white;
            box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3);
        }

        .btn-analyze:hover {
            background: #2563eb;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }

        .btn-analyze:active {
            transform: translateY(0);
        }

        .btn-clear {
            background: #ffffff;
            color: #64748b;
            border: 2px solid #e2e8f0;
        }

        .btn-clear:hover {
            background: #f8fafc;
            border-color: #cbd5e1;
            transform: translateY(-2px);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }

        /* Loading Animation */
        .loading {
            display: none;
            text-align: center;
            margin-top: 45px;
            padding: 40px;
            background: #f8fafc;
            border-radius: 16px;
            border: 2px solid #e2e8f0;
        }

        .loading.show {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        .spinner {
            width: 56px;
            height: 56px;
            border: 4px solid #e2e8f0;
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            font-size: 1.05em;
            color: #64748b;
            font-weight: 600;
            letter-spacing: 0.3px;
        }

        /* Result Box */
        .result-box {
            margin-top: 45px;
            padding: 45px;
            border-radius: 16px;
            display: none;
            border: 2px solid;
        }

        .result-box.show {
            display: block;
            animation: slideUp 0.5s ease;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .result-box.real {
            background: #f0fdf4;
            border-color: #86efac;
        }

        .result-box.fake {
            background: #fef2f2;
            border-color: #fca5a5;
        }

        .result-content {
            position: relative;
            z-index: 1;
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 35px;
            padding-bottom: 30px;
            border-bottom: 2px solid;
        }

        .result-box.real .result-header {
            border-bottom-color: #bbf7d0;
        }

        .result-box.fake .result-header {
            border-bottom-color: #fecaca;
        }

        .result-title {
            font-size: 2.4em;
            font-weight: 800;
            letter-spacing: -1px;
        }

        .result-title.real {
            color: #16a34a;
        }

        .result-title.fake {
            color: #dc2626;
        }

        .result-icon {
            font-size: 4em;
        }

        .confidence-section {
            margin-top: 30px;
        }

        .confidence-label {
            font-weight: 600;
            margin-bottom: 16px;
            color: #1e293b;
            font-size: 1.05em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .confidence-value {
            font-size: 1.5em;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
            color: #3b82f6;
        }

        .progress-bar {
            width: 100%;
            height: 40px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 1em;
            transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        /* Info Section */
        .info-section {
            margin-top: 40px;
            padding: 30px;
            background: #eff6ff;
            border-radius: 12px;
            border-left: 4px solid #3b82f6;
        }

        .info-title {
            font-size: 1.1em;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .info-text {
            color: #475569;
            font-size: 0.95em;
            line-height: 1.7;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 60px;
            padding-top: 40px;
            border-top: 2px solid #e2e8f0;
        }

        .footer-content {
            color: #64748b;
            font-size: 0.9em;
            font-weight: 500;
            line-height: 1.8;
        }

        .footer-links {
            margin-top: 20px;
            display: flex;
            gap: 30px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .footer-link {
            color: #3b82f6;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s ease;
        }

        .footer-link:hover {
            color: #2563eb;
            text-decoration: underline;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: 30px 20px;
            }

            .header {
                padding: 40px 0 30px;
            }

            .header h1 {
                font-size: 2.2em;
                letter-spacing: -1.5px;
            }

            .header .subtitle {
                font-size: 1em;
            }

            .logo-container {
                width: 70px;
                height: 70px;
                margin-bottom: 20px;
            }

            .logo {
                font-size: 2.5em;
            }

            .main-card {
                padding: 30px 20px;
            }

            .button-group {
                flex-direction: column;
            }

            .result-header {
                flex-direction: column;
                text-align: center;
                gap: 20px;
            }

            .result-title {
                font-size: 1.8em;
            }

            .result-icon {
                font-size: 3em;
            }

            .stats-grid {
                grid-template-columns: 1fr;
                gap: 16px;
            }

            .badge-container {
                flex-direction: column;
                align-items: center;
            }

            .footer-links {
                flex-direction: column;
                gap: 12px;
            }
        }
    </style>
</head>
<body>
    <div class="background"></div>

    <div class="container">
        <div class="header">
            <div class="logo-container">
                <div class="3️⃣🅿️"></div>
            </div>
            <h1>Fake News Detection System</h1>
            <h1>3PIXELS</h1>
            <h5>MEMBERS - Nayan Ghimire , Sumit Maharjan , Shashwot Jargha </h5>
            <p class="subtitle">Enterprise-Grade AI Technology for News Verification</p>
            <div class="badge-container">
                <span class="badge primary"> Hackathon Project</span>
                <span class="badge"> Machine Learning</span>
                <span class="badge"> Real-Time Analysis</span>
            </div>
        </div>

        {% if accuracy %}
        <div class="stats-card">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">OPERATIONAL</div>
                    <div class="stat-label">System Status</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.1f"|format(accuracy * 100) }}%</div>
                    <div class="stat-label">Model Accuracy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">ADVANCED</div>
                    <div class="stat-label">AI Architecture</div>
                </div>
            </div>
        </div>
        {% endif %}

        <div class="main-card">
            <div class="card-header">
                <h2 class="card-title">Article Analysis</h2>
                <p class="card-description">Submit a news article below for comprehensive authenticity verification using our state-of-the-art natural language processing model.</p>
            </div>

            <form id="newsForm">
                <div class="form-group">
                    <label for="newsText">News Article Content</label>
                    <textarea 
                        id="newsText" 
                        name="newsText" 
                        rows="12" 
                        placeholder="Enter the full text of the news article you wish to verify. Our AI model will analyze linguistic patterns, semantic coherence, and credibility indicators to determine authenticity..."
                        required
                    ></textarea>
                    <div class="char-count" id="charCount">0 characters</div>
                </div>

                <div class="button-group">
                    <button type="submit" class="btn-analyze">Analyze Article</button>
                    <button type="button" class="btn-clear" onclick="clearForm()">Clear Form</button>
                </div>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div class="loading-text">Processing article through neural network...</div>
            </div>

            <div class="result-box" id="resultBox">
                <div class="result-content">
                    <div class="result-header">
                        <div class="result-title" id="resultTitle"></div>
                        <div class="result-icon" id="resultIcon"></div>
                    </div>
                    <div class="confidence-section">
                        <div class="confidence-label">
                            <span>Model Confidence Score</span>
                            <span class="confidence-value" id="confidenceValue">0%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="confidenceFill" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="info-section">
                <div class="info-title">ℹ About This System</div>
                <p class="info-text">
                    This fake news detection system employs advanced machine learning algorithms trained on extensive datasets to identify patterns indicative of misinformation. The model analyzes multiple linguistic and semantic features to provide accurate classification results with quantified confidence levels.
                </p>
            </div>
        </div>

        
        </div>
    </div>

    <script>
        const newsText = document.getElementById('newsText');
        const charCount = document.getElementById('charCount');
        const newsForm = document.getElementById('newsForm');
        const loading = document.getElementById('loading');
        const resultBox = document.getElementById('resultBox');

        newsText.addEventListener('input', () => {
            const length = newsText.value.length;
            charCount.textContent = length.toLocaleString() + ' characters';
        });

        newsForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const text = newsText.value.trim();
            if (!text) return;

            loading.classList.add('show');
            resultBox.classList.remove('show');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: 'newsText=' + encodeURIComponent(text)
                });

                const data = await response.json();
                
                setTimeout(() => {
                    displayResult(data);
                }, 500);
            } catch (error) {
                alert('Error analyzing news. Please try again.');
                console.error(error);
            } finally {
                setTimeout(() => {
                    loading.classList.remove('show');
                }, 500);
            }
        });

        function displayResult(data) {
            const resultTitle = document.getElementById('resultTitle');
            const resultIcon = document.getElementById('resultIcon');
            const confidenceFill = document.getElementById('confidenceFill');
            const confidenceValue = document.getElementById('confidenceValue');

            resultBox.className = 'result-box show ' + data.prediction.toLowerCase();
            resultTitle.className = 'result-title ' + data.prediction.toLowerCase();
            resultTitle.textContent = 'CLASSIFIED AS ' + data.prediction.toUpperCase();
            
            resultIcon.textContent = data.prediction === 'Real' ? '✅' : '❌';
            
            confidenceValue.textContent = data.confidence.toFixed(1) + '%';
            
            setTimeout(() => {
                confidenceFill.style.width = data.confidence + '%';
            }, 100);
        }

        function clearForm() {
            newsText.value = '';
            charCount.textContent = '0 characters';
            resultBox.classList.remove('show');
            
            const confidenceFill = document.getElementById('confidenceFill');
            confidenceFill.style.width = '0%';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get('newsText', '')
    result = predict_news(news_text)
    return jsonify(result)

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" FAKE NEWS DETECTOR - STARTING UP")
    print("="*60 + "\n")
    
    train_model()
    
    print("\n" + "="*60)
    print(" WEB SERVER STARTING...")
    print("HAVE A PATIENCE!!!!!!")
    print("="*60)
    print("\n Open your browser and go to: http://127.0.0.1:5000")
    print("\n Press CTRL+C to stop the server\n")
    app.run(debug=True, port=5000, use_reloader=False)
