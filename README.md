# 🛡️ AI-Based Phishing Email Detection

This project uses **Machine Learning (Logistic Regression)** and **Natural Language Processing (TF-IDF)** to classify emails as *Phishing* or *Safe*.

---

## ⚙️ How It Works
1. **phishing_model_training.py** — trains a model using the dataset `phishing.csv` (not uploaded due to size).
2. **phishing_detector.py** — tests predictions with sample emails.
3. **app.py** — Streamlit web app for interactive detection.

---

## 🧠 Model
The model is trained using:
- TF-IDF Vectorization for feature extraction  
- Logistic Regression for classification  
- Accuracy: ~97 %

---

## 📂 Dataset
The dataset `phishing.csv` is too large to upload to GitHub.  
You can download a similar dataset from **[Kaggle – Phishing Email Detection Dataset](https://www.kaggle.com/)**.

Place the file in your project directory before running the training script.

---

## 🚀 Running Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/phishing-email-detector.git
   cd phishing-email-detector
