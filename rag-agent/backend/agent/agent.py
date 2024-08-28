import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor
from langchain import hub
from chains.course_info_vector_chain import course_info_vector_chain
from chains.NLP_and_DL_vector_chain import NLP_and_DL_vector_chain
from chains.You_tube_vector_chain import You_tube_vector_chain
from backend.agent.tools.factorial_fibonacci import (
    get_factorial,
    get_matrix_fibonacci,
)

CS224N_AGENT_MODEL = os.getenv("CS224N_AGENT_MODEL")

CS224N_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

# CS224N_agent_prompt = hub.pull("hwchase17/openai-functions-agent")
'''
https://smith.langchain.com/hub/hwchase17/openai-functions-agent
'''

tools = [
    Tool(
        name="Course Info",
        func=course_info_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about course itself for example Instructors, Course Manager, 
        Teaching Assistants, Logistics, Previous offerings, Schedule, 
        Reference Texts, Coursework, Course Materials etc.
        Use the entire prompt as input to the tool.
        For instance, if the prompt is:
        "Is Chris Manning Instructor of CS224N Stanford course?", 
        the input should be:
        "Is Chris Manning Instructor of CS224N Stanford course?"
        """,
    ),
    Tool(
        name="NLP and DL materials",
        func=NLP_and_DL_vector_chain.invoke,
        description="""Useful for answering questions about any type of questions about 
        Natural Language Processing(NLP), Deep Learning(DL), Large Language models(LLMs), 
        Mashine Learning(ML). 
        For example questions about bag of words, embedings, word2vec, RNN, LSTM, Transformers, 
        attention, self-attention and any others questions about and around NLP, DL, LLMs, ML. 
        Use the entire prompt as input to the tool.
        For instance, if the prompt is:
        'What is Transformer?"
        the input should be:
        "Is Chris Manning Instructor of CS224N Stanford course?"
        'What is Transformer?"
        """,
    ),
    Tool(
        name="You tube lectures",
        func=You_tube_vector_chain,
        description="""Use when asked about YouTube lectures... 
        ......
        """,
    ),
    Tool(
        name="Factorial",
        func=get_factorial,
        description="""
        Purpose: Use this tool to calculate the exact factorial value of a given integer.
        Instructions: 
        1. Input only the integer for which the factorial needs to be calculated. 
        Do not include any additional words or phrases like "factorial of."
        2. The tool will return a single number, which is the factorial of the input integer.
        """,
    ),
        Tool(
        name="Fibonachi",
        func=get_matrix_fibonacci,
        description="""
        Use when you need to find out exect value of fibonacci of the integer number.
        Do not pass the words "fibonacci of" as input, only the exectly integer number 
        of the factorial which value neeed to evalute. 
        This tool returns a one number which is fibonacci of number N.
        """,
    ),
]

chat_model = ChatOpenAI(
    model = CS224N_AGENT_MODEL,
    temperature=0,
)

CS224N_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=CS224N_agent_prompt,
    tools=tools,
)

CS224N_rag_agent_executor = AgentExecutor(
    agent=CS224N_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
