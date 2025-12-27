import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (needed for Streamlit Cloud)
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("toxic_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

st.set_page_config(page_title="Toxic Comment Detection", page_icon="🚫")

st.title("🚫 Real-Time Toxic Comment Detection")
st.write("Enter a comment below to check whether it is toxic or non-toxic.")

user_input = st.text_area("Enter a comment:")

if st.button("Check Toxicity"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean = clean_text(user_input)
        vector = tfidf.transform([clean])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("⚠️ Toxic Comment Detected")
        else:
            st.success("✅ Comment is Non-Toxic")
