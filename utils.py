import re
import joblib
import nltk
from nltk.corpus import stopwords

# ==========================================
# DOWNLOAD STOPWORDS
# ==========================================

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/final_model.pkl")
vectorizer = joblib.load("models/final_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):

    text = str(text)

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = text.lower()

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

# ==========================================
# PREDICT NEWS
# ==========================================

def predict_news(news):

    cleaned = clean_text(news)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(vector)[0]) * 100
    else:
        confidence = 95.0

    return prediction, confidence