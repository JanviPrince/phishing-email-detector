import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="AI Phishing Email Detector", page_icon="🛡️")

st.title("🛡️ AI-Powered Phishing Email Detector")
st.write("Paste an email below and let AI detect if it's a **Phishing** or **Safe** email.")

# Input area
email_input = st.text_area("✉️ Enter the email content:", height=200)

if st.button("🔍 Analyze Email"):
    if email_input.strip() == "":
        st.warning("Please enter an email text first.")
    else:
        # Transform and predict
        email_vector = vectorizer.transform([email_input])
        prediction = model.predict(email_vector)[0]
        label = "🚨 Phishing Email" if prediction == "Phishing Email" else "✅ Safe Email"
        st.subheader(f"Result: {label}")
