
# AI Agents, RAGs, and Chatbots

This repository contains my projects focused on:

**AI Agents**, **Retrieval-Augmented Generation (RAG)**, and **AI Chatbots**.

### 1. [Async AI Bot](https://github.com/VladGKulikov/AI-Agents-RAGs-and-Chat-Bots/tree/main/async_ai_bot)  
(Python, JS, html, asyncio, FastAPI, WebSockets)

Asynchronous AI bot without any specific UI or AI frameworks, but with memory capabilities, separate dialogue threads, FastAPI, a custom web interface (index.html), and WebSockets.

---

### 2. [RAG-Bot with multiple RAG methods](https://github.com/VladGKulikov/AI-Agents-RAGs-and-Chat-Bots/tree/main/rag-bot-langchain-fastapi-sreamlit)  
(LangChain, FastAPI, Streamlit, Chroma Vector DB, Docker)

This project implements a Retrieval-Augmented Generation (RAG) system. It uses FastAPI for the backend, Streamlit for the frontend, and LangChain with ChromaDB for RAG functionality. Docker/Docker-Compose is used for convenient deployment. 

The project features multiple RAG methods with interactive switching between them.

---

### 3. [RAG with LangChain](https://github.com/VladGKulikov/AI-Agents-RAGs-and-Chat-Bots/tree/main/RAG_witn_LangChain)  
(LangChain, FastAPI, Streamlit, Chroma Vector DB, Docker)

An intermediate project, operational but with partially implemented functionality. The plan is to assess the quality of the RAG system by creating a synthetic dataset of questions and answers generated by a large LLM, evenly covering the original data. This large LLM will also evaluate the responses of a smaller model in the RAG system, generating an accuracy score to measure the performance. Though specific frameworks exist for RAG quality assessment, the goal of this project is to provide a rapid preliminary evaluation.

---

### 4. [RAG Agent with multiple RAG Tools](https://github.com/VladGKulikov/AI-Agents-RAGs-and-Chat-Bots/tree/main/rag-agent)  
(LangChain, FastAPI, Streamlit, Chroma Vector DB, Docker)

This project extends the RAG system by incorporating specialized tools for mathematical operations, where LLMs typically struggle. The RAG Agent can answer questions from a PDF file and perform computations like factorials and Fibonacci sequences for large values (up to 50,000 digits). For instance, you can ask the agent, "What is the factorial of 1,000?" or "Fibonacci 1,000?" and the correct answer will be generated using the appropriate tool.  

This project uses the now-legacy LangChain AgentExecutor interface.  
An updated version, using the LangGraph system, can be found in my other project: **LangGraph-Self-RAG-Agent**.

---

### 5. [LangGraph-Self-RAG-Agent](https://github.com/VladGKulikov/AI-Agents-RAGs-and-Chat-Bots/tree/main/LangGraph-Self-RAG-Agent)  
(LangGRAPH, LangChain, FastAPI, Streamlit, Chroma Vector DB, Docker)

This project implements a self-reflective AI RAG Agent system. It includes mechanisms for self-reflection and self-grading on both retrieved documents and generated responses. If the generated answer is deemed unsatisfactory or contains hallucinations, the original query is rewritten, and the process repeats until a satisfactory result is achieved, or a maximum iteration limit is reached.  
The code builds upon ideas from the [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/) with my own modifications and interpretations.    
The core concept is inspired by the October 2023 paper: **[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/)**.

