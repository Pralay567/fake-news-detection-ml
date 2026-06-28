#Imports
import re
import warnings
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

#Download Stopwords
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

#Load Datasets
print("Loading datasets...")

fake_df = pd.read_csv("datasets/fake.csv")
true_df = pd.read_csv("datasets/true.csv")

fake_df["label"] = 1
true_df["label"] = 0

fake_df["full_text"] = (
    fake_df["title"].fillna("") + " " +
    fake_df["text"].fillna("")
)

true_df["full_text"] = (
    true_df["title"].fillna("") + " " +
    true_df["text"].fillna("")
)

main_df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)

main_df = main_df[["full_text", "label"]] 

#Load IFND Dataset
try:

    print("Loading IFND dataset...")

    ifnd = pd.read_csv(
        "datasets/IFND.csv",
        encoding="latin1",
        on_bad_lines="skip"
    )

    if "Statement" in ifnd.columns:
        ifnd.rename(
            columns={"Statement": "full_text"},
            inplace=True
        )

    if "Label" in ifnd.columns:

        ifnd["label"] = ifnd["Label"].apply(
            lambda x:
            0 if str(x).upper() in
            ["REAL", "TRUE", "0"]
            else 1
        )

    ifnd = ifnd[["full_text", "label"]]

    main_df = pd.concat(
        [main_df, ifnd],
        ignore_index=True
    )

    print("✅ IFND added")

except Exception as e:

    print(f"Skipping IFND: {e}")

#Clean Dataset
main_df.dropna(inplace=True)
main_df.drop_duplicates(inplace=True)

main_df = main_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(f"Dataset size: {len(main_df)}")

# ==========================================
# TEXT PREPROCESSING
# ==========================================

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep only letters and numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

print("Cleaning text...")

main_df["clean_text"] = main_df["full_text"].apply(clean_text)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X = main_df["clean_text"]
y = main_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ==========================================
# TF-IDF VECTORIZER
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.85
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorization completed.")

# ==========================================
# ENSEMBLE MODEL
# ==========================================

print("Building ensemble model...")

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

nb = MultinomialNB()

svc = LinearSVC(
    random_state=42
)

svc = CalibratedClassifierCV(
    svc,
    cv=3
)

model = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("nb", nb),
        ("svc", svc)
    ],
    voting="soft"
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("Training model...")

model.fit(
    X_train_vec,
    y_train
)

predictions = model.predict(
    X_test_vec
)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 50)
print(f"Accuracy : {accuracy * 100:.2f}%")
print("=" * 50)

print("\nClassification Report\n")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix\n")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/final_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/final_vectorizer.pkl"
)

print("\n✅ Model saved successfully!")
print("📁 models/final_model.pkl")
print("📁 models/final_vectorizer.pkl")
