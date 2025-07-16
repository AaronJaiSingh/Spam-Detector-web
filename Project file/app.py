import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Web interface
st.title("📩 Spam Message Detector")
st.write("Enter a message and the model will predict whether it's **SPAM** or **HAM**.")

# Input field
user_input = st.text_area("Enter your message:")

# Predict
if st.button("Predict"):
    if user_input.strip() != "":
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)[0]
        result = "🛑 SPAM" if prediction == 1 else "✅ HAM"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter a message to predict.")
