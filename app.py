# VyomaSecurity Spam Checker (All-in-One)
# Author: Prasoon Jadon | Vyoma Labs
# Purpose: Educational & Defensive Cybersecurity Use Only

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -----------------------------
# Config
# -----------------------------
DATASET_PATH = "spam_dataset.csv"
MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

app = Flask(__name__)

# -----------------------------
# Train Model (if not exists)
# -----------------------------
def train_model():
    data = pd.read_csv(DATASET_PATH)

    data["label"] = data["label"].map({"spam": 1, "ham": 0})

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    print(f"[+] Model trained with accuracy: {accuracy:.2f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)


# Train only once
if not os.path.exists(MODEL_PATH):
    print("[*] Training spam detection model...")
    train_model()

# Load model
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -----------------------------
# Routes
# -----------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "VyomaSecurity Spam Checker",
        "status": "running",
        "ethics": "No messages are stored or logged"
    })


@app.route("/check-spam", methods=["POST"])
def check_spam():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Text is required"}), 400

    text = data["text"]

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0][prediction]

    return jsonify({
        "input": "received",
        "result": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(float(probability), 3)
    })


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
