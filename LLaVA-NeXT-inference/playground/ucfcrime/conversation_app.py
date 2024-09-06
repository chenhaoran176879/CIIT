import streamlit as st
import requests

API_URL = "http://localhost:8501/generate/"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.title("Qwen-1.5 Chat Interface")

chat_history = st.container()

with chat_history:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You**: {message['content']}")
        else:
            st.markdown(f"**Assistant**: {message['content']}")


prompt_input = st.text_area("Prompt:", placeholder="Enter your prompt...", height=100)
current_question_input = st.text_area("Question:", placeholder="Ask your question...", height=100)


send_button = st.button("Send")

if send_button and (prompt_input or current_question_input):
    user_input = f"Prompt: {prompt_input}\n\nQuestion: {current_question_input}"
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    response = requests.post(API_URL, json={"messages": st.session_state.messages})
    assistant_message = response.json().get("response")
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    
    st.experimental_set_query_params(rerun="true")

if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.experimental_set_query_params(rerun="true")
