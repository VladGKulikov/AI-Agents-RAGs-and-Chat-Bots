from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import RAGSystem, RAGConfig
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

config = RAGConfig(
    model=os.getenv('MODEL'),
    embed_model=os.getenv('EMBED_MODEL'),
    chroma_db_path=os.getenv('CHROMA_DB_PATH')
)

rag_system = RAGSystem(config)

class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_question(question: Question):
    try:
        answer = rag_system.simple_rag(question.text)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_multi_query")
async def ask_question_multi_query(question: Question):
    try:
        answer = rag_system.multi_query_rag(question.text)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_rag_fusion")
async def ask_question_rag_fusion(question: Question):
    try:
        answer = rag_system.rag_fusion(question.text)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
