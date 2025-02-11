import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import base64
from io import BytesIO

API_URL = "http://127.0.0.1:5000"


def fetch_articles():
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            return response.json()
        st.error(f"Error fetching articles: {response.text}")
        return []
    except requests.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return []


def summarize_text(text):
    try:
        response = requests.post(f"{API_URL}/summarize", json={"text": text})
        if response.status_code == 200:
            return response.json()
        st.error(f"Error getting summary: {response.text}")
        return None
    except requests.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def plot_from_base64(base64_string):
    if base64_string:
        st.image(BytesIO(base64.b64decode(base64_string)))


st.title("News Summarization & Sentiment Analysis")

# Text input area
text_input = st.text_area("Enter text to summarize:", height=200)
if st.button("Summarize"):
    if text_input:
        with st.spinner("Huggingface Model summaries..."):
            summary_data = summarize_text(text_input)
            if summary_data:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Huggingface Model Summary")
                    st.write(summary_data['generic_summary'])
                    st.subheader("Sentiment Analysis")
                    plot_from_base64(summary_data['generic_plot'])

                with col2:
                    st.subheader("Fine-tuned Model Summary")
                    st.write(summary_data['fine_tuned_summary'])
                    st.subheader("Sentiment Analysis")
                    plot_from_base64(summary_data['fine_tuned_plot'])
    else:
        st.warning("Please enter some text to summarize.")

# Display saved articles
st.subheader("Recent Articles")
articles = fetch_articles()
if articles:
    for article in articles:
        with st.expander(article['title']):
            st.write(article['content'])
            if st.button(f"Summarize", key=article['id']):
                summary_data = summarize_text(article['content'])
                if summary_data:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Huggingface Model Summary")
                        st.write(summary_data['generic_summary'])
                        st.subheader("Sentiment Analysis")
                        plot_from_base64(summary_data['generic_plot'])

                    with col2:
                        st.subheader("Fine-tuned Model Summary")
                        st.write(summary_data['fine_tuned_summary'])
                        st.subheader("Sentiment Analysis")
                        plot_from_base64(summary_data['fine_tuned_plot'])
