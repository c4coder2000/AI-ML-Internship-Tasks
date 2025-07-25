import streamlit as st
from infer import predict

st.set_page_config(page_title="News Classifier", layout="centered")
st.title("📰 BERT News Headline Classifier")

text = st.text_input("Enter a news headline:")
if text:
    label = predict(text)
    st.success(f"Predicted Category: **{label}**")
