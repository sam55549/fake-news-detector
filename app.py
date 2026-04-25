import streamlit as st
import pickle
st.write("App is running...")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection App")

news_input = st.text_area("Enter News Text:")

def predict_news(news):
    vect = vectorizer.transform([news])
    prediction = model.predict(vect)
    return prediction[0]

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_news(news_input)

        if result == 0:
            st.error("Fake News ❌")
        else:
            st.success("Real News ✅")