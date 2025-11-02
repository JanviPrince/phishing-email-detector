# phishing_detector.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load dataset
data = pd.read_csv("Phishing_Email.csv")

print("✅ Dataset loaded successfully!")
print("Rows:", data.shape[0])
print("Columns:", list(data.columns))

# 2️⃣ Check and clean missing data
data = data.dropna()

# 3️⃣ Separate features (text) and labels
X = data['Email Text']
y = data['Email Type']
   # phishing or legitimate

# 4️⃣ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Convert text into numeric vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6️⃣ Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 7️⃣ Predict and evaluate
y_pred = model.predict(X_test_tfidf)

print("\n✅ Model Training Complete!")
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 8️⃣ Try with your own example
sample_email = [
    "Your account has been suspended. Click here to verify your identity.",
    "Hi team, please find attached the meeting agenda for tomorrow."
]

sample_tfidf = vectorizer.transform(sample_email)
predictions = model.predict(sample_tfidf)

for email, pred in zip(sample_email, predictions):
    print("\nEmail:", email)
    print("Prediction:", pred)
