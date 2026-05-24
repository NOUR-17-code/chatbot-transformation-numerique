import streamlit as st

st.set_page_config(page_title="Chatbot", page_icon="🤖")

st.title("Chatbot Transformation Numérique")

question = st.text_input("Posez votre question")

if question:
    st.success(f"Votre question : {question}")
