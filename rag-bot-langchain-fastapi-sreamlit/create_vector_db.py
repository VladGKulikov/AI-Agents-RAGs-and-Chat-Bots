from tqdm import tqdm
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
# import bs4
# from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

MODEL = os.getenv('MODEL') 
EMBED_MODEL = os.getenv('EMBED_MODEL')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(model=EMBED_MODEL) 


def create_persistent_chroma_db(splits, embedding, persist_directory):
    vectorstore = None

    # Create a progress bar
    with tqdm(total=len(splits), desc="Creating Index", unit="doc") as pbar:
        # Iterate through the documents
        for i, document in enumerate(splits):
            # Create or update the vector store with each document            
            if vectorstore is None: 
                vectorstore = Chroma.from_documents([document], 
                                                    embedding=embedding, 
                                                    persist_directory=persist_directory
                                                    )
            else:
                vectorstore.add_documents([document])

            # Update the progress bar
            pbar.update(1)

    return vectorstore

# Jurafsky and Martin Chapter 10 (Transformers and Large Language Models)
source = 'https://web.stanford.edu/~jurafsky/slpdraft/10.pdf'

# Specify the path to the vectorstore directory
chroma_directory_path = './vector-db/' # './vector-db/ChrDb'+'_' + EMBED_MODEL

# Check if the directory exists
# if not os.path.isdir(chroma_directory_path):
# Load and Split
loader = PyPDFLoader(source) 
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f'Splited on the {len(splits)} parts')    

# Index
vectorstore = create_persistent_chroma_db(splits, 
                                        embedding=embedding, 
                                        persist_directory=chroma_directory_path
                                        )
    