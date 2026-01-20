import streamlit as st
import pickle
import pandas as pd
import re
import string
from nltk.corpus import stopwords
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text
st.title("✈️ Airline Tweet Sentiment Analysis")
st.write("Enter a tweet to analyze its sentiment.")
st.divider()
user_input = st.text_area(
    "Enter tweet text",
    placeholder="e.g. Flight was delayed for hours and staff was rude",
    height=120
)
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = float(max(probability))   # ✅ THIS FIXES YOUR ERROR

        if prediction == 1:
            st.success(f"😊 Positive (Confidence: {confidence:.2f})")
        elif prediction == 0:
            st.info(f"😐 Neutral (Confidence: {confidence:.2f})")
        else:
            st.error(f"😠 Negative (Confidence: {confidence:.2f})")

        # Confidence indicator
        if confidence > 0.8:
            st.caption("🔴 High confidence prediction")
        elif confidence > 0.6:
            st.caption("🟠 Medium confidence prediction")
        else:
            st.caption("🟡 Low confidence prediction")
st.divider()
st.subheader("📈 Dataset Sentiment Distribution")

@st.cache_data
def load_data():
    return pd.read_csv("sentiment.csv")

data = load_data()
st.bar_chart(data['airline_sentiment'].value_counts())
st.divider()
st.markdown("<center>Built with ❤️ using ML, NLP & Streamlit</center>", unsafe_allow_html=True)


