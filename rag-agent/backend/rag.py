import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dataclasses import dataclass
# from typing import List, Dict, Any
from langchain.load import dumps, loads

load_dotenv()

@dataclass
class RAGConfig:
    model: str
    embed_model: str
    chroma_db_path: str

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

    def _create_multi_query_rag_chain(self):
        multi_query_template = """You are an AI language model assistant. Your task is to generate three 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""

        prompt_perspectives = ChatPromptTemplate.from_template(multi_query_template)

        generate_queries = (
            prompt_perspectives 
            | self.llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )        

        def get_unique_union(documents):
            """ Unique union of retrieved docs """
            # Flatten list of lists, and convert each Document to string
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            # Get unique documents
            unique_docs = list(set(flattened_docs))
            # Return
            return [loads(doc) for doc in unique_docs]

        retrieval_chain = generate_queries | self.retriever.map() | get_unique_union

        multi_query_rag_chain = (
            {"context": retrieval_chain, 
             "question": itemgetter("question")}
            | self.simple_RAG_prompt
            | self.llm
            | StrOutputParser()
        )

        return multi_query_rag_chain

    def _create_rag_fusion_chain(self):
        rag_fusion_template = """You are a helpful assistant that generates multiple search queries based on a single input query.
        Generate multiple search queries related to: {question}
        Output (3 queries):"""

        prompt_rag_fusion = ChatPromptTemplate.from_template(rag_fusion_template)

        generate_queries = (
            prompt_rag_fusion 
            | self.llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )        

        def reciprocal_rank_fusion(results, k=60):
            """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
                and an optional parameter k used in the RRF formula """
            
            # Initialize a dictionary to hold fused scores for each unique document
            fused_scores = {}

            # Iterate through each list of ranked documents
            for docs in results:
                # Iterate through each document in the list, with its rank (position in the list)
                for rank, doc in enumerate(docs):
                    # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                    doc_str = dumps(doc)
                    # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    # Retrieve the current score of the document, if any
                    previous_score = fused_scores[doc_str]
                    # Update the score of the document using the RRF formula: 1 / (rank + k)
                    fused_scores[doc_str] += 1 / (rank + k)

            # Sort the documents based on their fused scores in descending order to get the final reranked results
            reranked_results = [
                (loads(doc), score)
                    for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]


            # Return the reranked results as a list of tuples, each containing the document and its fused score
            return reranked_results
        
        retrieval_chain_rag_fusion = generate_queries | self.retriever.map() | reciprocal_rank_fusion

        rag_fusion_chain = (
            {"context": retrieval_chain_rag_fusion, 
             "question": itemgetter("question")} 
            | self.simple_RAG_prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_fusion_chain
    
    def simple_rag(self, question):
        return self._create_simple_rag_chain().invoke(question)

    def multi_query_rag(self, question):
        return self._create_multi_query_rag_chain().invoke({"question":question})

    def rag_fusion(self, question):
        return self._create_rag_fusion_chain().invoke({"question":question})       
    