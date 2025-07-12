import streamlit as st
import joblib
model = joblib.load("FAKE-NEWS-DETECTION/fake_news_model.pkl")
vectorizer = joblib.load("FAKE-NEWS-DETECTION/tfidf_vectorizer.pkl")
st.title("📰 Fake News Detection App")
user_input = st.text_area("Enter a news article or headline:")
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)
        result = "🔴 Fake News" if prediction[0] == 0 else "🟢 Real News"
        st.subheader("Prediction:")
        st.success(result)
