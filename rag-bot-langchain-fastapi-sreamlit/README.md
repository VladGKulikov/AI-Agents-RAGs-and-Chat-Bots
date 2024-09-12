# RAG Application with FastAPI, Streamlit, and LangChain

This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI for the backend, Streamlit for the frontend, and LangChain with ChromaDB for the RAG functionality.

This is a demo based as a source for RAG on the [PDF file](https://web.stanford.edu/~jurafsky/slpdraft/10.pdf) from the book "Speech and Language Processing" by Jurafsky and Martin, 
specifically Chapter 10, Transformers and Large Language Models. 
The link on pdf-source is from Stanford's Spring 2024 CS224N "Natural Language Processing with Deep Learning course". 

![alt text](Img/image-1.png)

## Prerequisites

- Docker
- Docker Compose

## Setup

1. Clone this repository.
2. Navigate to the project root directory.
3. Create a `.env` file in the `backend` directory with your OpenAI API key and other configurations:

For example:
```
OPENAI_API_KEY=your_openai_api_key
USER_AGENT=DefaultLangchainUserAgent
MODEL='gpt-4o-mini'
EMBED_MODEL='text-embedding-3-small'
CHROMA_DB_PATH='vector-db'
LARGE_EVALUATING_MODEL='gpt-4o'
```

4. Make sure your vector database is present in the `vector-db` directory.
The database should be located directly in this directory.
You can use create_vector_db.py with a few small changes.

## Running the Application

1. Build and start the containers:

```
docker-compose up --build
```

2. Access the Streamlit frontend at `http://localhost:8501`.
3. Enter your questions in the text input and click "Ask" to get answers from the RAG system.

## Project Structure

- `backend/`: Contains the FastAPI backend and RAG implementation.
- `frontend/`: Contains the Streamlit frontend.
- `docker-compose.yml`: Defines the multi-container Docker application.
- `backend/vector-db/`: Directory for storing the ChromaDB vector database.

## Customization
- To switch between different RAG methods, select method in UI selectbox.
- To switch between different models, update the `MODEL` variable in the `.env` file.
- To modify the RAG implementation, edit the `rag.py` file in the backend directory.
