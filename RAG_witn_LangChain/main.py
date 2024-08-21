import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

LARGE_EVALUATING_MODEL = os.getenv('LARGE_EVALUATING_MODEL')
MODEL = os.getenv('MODEL')
EMBED_MODEL = os.getenv('EMBED_MODEL')
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")        

# print(f'MODEL={MODEL}')
# input()
llm = None
embedding = OpenAIEmbeddings(model=EMBED_MODEL) 

match MODEL:
    case 'gpt-4o-mini':                
        llm = ChatOpenAI(model_name=MODEL, temperature=0)
    case 'llama3.1':
        llm = Ollama(model=MODEL)
        # embedding = OllamaEmbeddings(model=EMBED_MODEL)
    case _:
        llm = Ollama(model=MODEL)
        # embedding = OllamaEmbeddings(model=EMBED_MODEL)
              
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding)                                    

retriever = vectorstore.as_retriever()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
''' 
**************************************************
1. One question simple RAG
**************************************************
'''

# Simple prompt
simple_RAG_template = """Answer the question based on the context below. 
The length of your answer should be no more than 3 sentences.
If answer should be Yes or No - answer only Yes or No.

Context: {context}

Question: {question}
"""

simple_RAG_prompt = ChatPromptTemplate.from_template(simple_RAG_template)

simple_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | simple_RAG_prompt
    | llm
    | StrOutputParser()
)


evaluation_prompt = '''You are a powerful AI model tasked with evaluating the accuracy of another, less powerful AI model's answers to given questions.

For the evaluation, you have:
1. The original question (Question)
2. The correct reference answer (Target_answer)
3. The answer provided by the junior AI model (Real_answer)

Your task is to assess whether the answer given by the junior AI model matches the meaning of the reference answer. A match in meaning means that the Real_answer conveys the same core information as the Target_answer, even if synonyms are used or the phrasing is slightly different.

If the junior AI model's answer matches the meaning of the reference answer, return 1. If it does not match, return 0. The answer should be strictly numerical: 1 or 0.

Question: {question}

Target_answer: {target_answer}

Real_answer: {real_answer}
'''

# import pandas as pd
# q_a = pd.read_csv('questions-answers.csv')

# for row in q_a:
#     question, target_answer = row['Question'], row['Answer']



# questions = ['How name this chapter?', 
#              'What is Transformer',
#              'What is attention?', 
#              'What is self-attention',
#              'What are LLMs']

# print()
# for i, question in enumerate(questions):        
#     print(f'Q №{i+1}:\n{question}')    
#     print(f'Answer from model {MODEL}:')
#     print(rag_chain.invoke(question))
#     print()
    
    
# answers = rag_chain.batch([f'question:{q}' for q in questions])
# print(answers)

''' 
**************************************************
2. Multi Query RAG
**************************************************
'''
# Multi Query: Different Perspectives
multi_query_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(multi_query_template)

generate_queries = (
    prompt_perspectives 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
retrieval_chain = generate_queries | retriever.map() | get_unique_union

from operator import itemgetter

# RAG

multi_query_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | simple_RAG_prompt
    | llm
    | StrOutputParser()
)


''' 
**************************************************
3. RAG-Fusion
**************************************************
'''

# RAG-Fusion: Related
RAG_Fusion_template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(RAG_Fusion_template)

generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
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

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion


rag_fusion_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | simple_RAG_prompt
    | llm
    | StrOutputParser()
)


''' 
**************************************************
3. Decomposition
**************************************************
'''




''' 
**************************************************
4. End
**************************************************
'''
questions = ['What is Transformer'] #'How name this chapter?', 'What is Transformer', 'Is pretraining a common step before fine-tuning transformers for specific tasks?']

print('-|-'*20)
for i, question in enumerate(questions):        
    print(f'Q №{i+1}:\n{question}')    
    print(f'Answer from model {MODEL}:')
    answer = simple_rag_chain.invoke(question)
    print(f'Simple_RAG:\n{answer}')    
    print()
    print(f'Multi Query RAG:\n{multi_query_rag_chain.invoke({"question":question})}')    
    print()
    print(f'RAG_Fusion:\n{rag_fusion_chain.invoke({"question":question})}')    
    print()
    print('-|-'*20)


