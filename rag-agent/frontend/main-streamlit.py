import streamlit as st
import requests

BACKEND_URL = "http://backend:8000/cs224n-rag-agent"

# st.output_text =''
# st.endpoint = ''

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        designed to answer questions about user's data.        
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
st.info("Ask me questions about [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/))")


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

    with st.spinner("Searching for an answer..."):
        print(f'Sending request with data: {data}')
        response = requests.post(BACKEND_URL, json=data)

        if response.status_code == 200:
            response_data = response.json()
            print(f'Received response: {response_data}')
            output_text = response_data["output"] # ["answer"]
            # explanation = response.json()["intermediate_steps"]
        else:
            print(f'Error response: {response.text}')
            output_text = f"An error occurred while processing your message. Please try again or rephrase your message. Status code: | {response.status_code} |  Data: {data} | Error: {response.text} |"


    st.chat_message("assistant").markdown(output_text)
    # st.status("How was this generated?", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            # "explanation": explanation,
        }
    )
