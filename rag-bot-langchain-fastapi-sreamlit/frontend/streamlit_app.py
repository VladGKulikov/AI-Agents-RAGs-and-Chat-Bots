import streamlit as st
import requests

BACKEND_URL = "http://backend:8000"

st.output_text =''
st.endpoint = ''

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        chat-bot designed to answer questions about user's data.
        The chat-bot uses retrieval-augment generation (RAG) over unstructured 
        data from pdf and html files.
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

st.title("Demo RAG System Chatbot")
st.info("Ask me questions about [Jurafsky and Martin Chapter 10 (Transformers and Large Language Models](https://web.stanford.edu/~jurafsky/slpdraft/10.pdf))")

# Add RAG method selection
rag_method = st.selectbox(
    "Select RAG method",
    ["Simple RAG", "Multi Query RAG", "RAG Fusion"],
    index=0,
    help="Choose the RAG method for question answering"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt) # , avatar = ":material/self_improvement:"
    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner(f"Searching for an answer using {rag_method}..."):
        # Choose the appropriate endpoint based on the selected RAG method
        if rag_method == "Simple RAG":
            st.endpoint = f"{BACKEND_URL}/ask"            
        elif rag_method == "Multi Query RAG":
            st.endpoint = f"{BACKEND_URL}/ask_multi_query"            
        else:  # RAG Fusion
            st.endpoint = f"{BACKEND_URL}/ask_rag_fusion"
        
        response = requests.post(st.endpoint, json=data)
        if response.status_code == 200:
            st.output_text = response.json()["answer"]
        else:
            st.output_text = f"""An error occurred while processing your message.
            Please try again or rephrase your message. Error: {response.text}"""

    # Display the current output_text
    if st.output_text:
        st.output_text += f'\nEndpoint: {st.endpoint}'+f'\nUsed RAG method: {rag_method}'
        st.chat_message("assistant").markdown(st.output_text)
        st.session_state.messages.append({"role": "assistant", "output": st.output_text})
        

    # Add a note about the current RAG method
    # st.markdown(f"endpoint: {endpoint}*")
    # st.markdown(f"*Used RAG method: {rag_method}*")