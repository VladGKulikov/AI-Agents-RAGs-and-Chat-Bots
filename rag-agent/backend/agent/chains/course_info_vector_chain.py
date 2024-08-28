import os
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from dataclasses import dataclass

load_dotenv()

@dataclass
class RAGConfig:
    model: str
    embed_model: str
    chroma_db_path: str

############# REWRITE #############

class RAGSystem:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = self._get_llm()
        self.embedding = OpenAIEmbeddings(model=self.config.embed_model)
        self.vectorstore = Chroma(persist_directory=self.config.chroma_db_path, 
                                  embedding_function=self.embedding)
        self.retriever = self.vectorstore.as_retriever()

        simple_RAG_template = """Answer the question based on the context below. 
        The length of your answer should be no more than 3 sentences.
        If answer should be Yes or No - answer only Yes or No.

        Context: {context}

        Question: {question}
        """
        self.simple_RAG_prompt = ChatPromptTemplate.from_template(simple_RAG_template)

    def _get_llm(self):
        if self.config.model == 'gpt-4o-mini':
            return ChatOpenAI(model_name=self.config.model, temperature=0)
        else:
            return Ollama(model=self.config.model)

    def _create_simple_rag_chain(self):
        simple_rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.simple_RAG_prompt
            | self.llm
            | StrOutputParser()
        )
        return simple_rag_chain