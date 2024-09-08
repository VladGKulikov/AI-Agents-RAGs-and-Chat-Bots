# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

import streamlit as st
import requests

BACKEND_URL = "http://backend:8000/self-rag-agent"

# st.output_text =''
# st.endpoint = ''

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This Self-RAG Agent with
        [LangGraph](https://langchain-ai.github.io/langgraph/)
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        is designed to answer questions about the user's data. 
        The code and idea are based on        
        [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/) and this [paper](https://arxiv.org/abs/2310.11511)
        
        """
    )

    st.header("Example Questions")
    st.markdown("- What is Transformer?")
    st.markdown("- What is attention?")
    st.markdown("- What is self-attention?")
    st.markdown("- What are transformer blocks?")
    st.markdown("- How do transformers achieve parallelism?")
    st.markdown("- Do transformers support autoregressive text generation?")
    st.markdown("- Are residual connections essential for the performance of transformers?")

st.title("Self-RAG Agent")
st.info(
    "Ask me questions about [Jurafsky and Martin Chapter 10 (Transformers and Large Language Models](https://web.stanford.edu/~jurafsky/slpdraft/10.pdf))")

if "messages" not in st.session_state:
    st.session_state.messages = []

avatar="🇦🇮"

for message in st.session_state.messages:

    avatar = "🇦🇮" if message["role"] == "assistant" else "👽" # "🇦🇮" "🤖" "👽"

    with st.chat_message(message["role"], avatar=avatar):
        if "output" in message.keys():
            st.markdown(message["output"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user", avatar="👽").markdown(prompt) 
    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        print(f'Sending request with data: {data}')
        response = requests.post(BACKEND_URL, json=data)

        if response.status_code == 200:
            response_data = response.json()
            print(f'Received response: {response_data}')
            output_text = response_data["output"]
        else:
            print(f'Error response: {response.text}')
            output_text = f"An error occurred while processing your message. Please try again or rephrase your message. Status code: | {response.status_code} |  Data: {data} | Error: {response.text} |"

    st.chat_message("assistant", avatar="🇦🇮").markdown(output_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text
        }
    )
