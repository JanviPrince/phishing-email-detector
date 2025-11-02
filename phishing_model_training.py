import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re, string
import joblib

# --- Load data ---
data = pd.read_csv(r"C:\Users\91628\Downloads\archive\Phishing_Email.csv")
data = data.dropna()
data.rename(columns={'Email Text': 'text', 'Email Type': 'label'}, inplace=True)

# --- Text cleaning ---
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", '', t)
    t = re.sub(r"\d+", '', t)
    t = t.translate(str.maketrans('', '', string.punctuation))
    return t.strip()

data['text'] = data['text'].apply(clean_text)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# --- Vectorize ---
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Train Logistic Regression (best model) ---
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Save model and vectorizer ---
joblib.dump(model, r"C:\Users\91628\Downloads\archive\phishing_model.pkl")
joblib.dump(vectorizer, r"C:\Users\91628\Downloads\archive\vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
