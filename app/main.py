import joblib
import os

# Get absolute path of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct model paths
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fraud_detection_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)


def predict_internship(text):
    vec = tfidf.transform([text])
    fraud_prob = model.predict_proba(vec)[0][1]

    if fraud_prob < 0.4:
        label = "Likely Genuine"
    elif fraud_prob < 0.7:
        label = "Suspicious"
    else:
        label = "High Scam Risk"

    return label, round(fraud_prob * 100, 2)


# Test
if __name__ == "__main__":
    text = input("Enter internship description: ")
    result, score = predict_internship(text)
    print("Prediction:", result)
    print("Fraud Probability:", score, "%")
