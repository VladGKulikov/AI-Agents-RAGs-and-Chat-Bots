from fastapi import FastAPI, HTTPException
from agent.agent import CS224N_rag_agent_executor
from pydantic import BaseModel


class CS224NQueryInput(BaseModel):
    text: str


class CS224NQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]


app = FastAPI(
    title="CS224N Stanford 'NLP and DL' course Chatbot",
    description="Endpoints for CS224N RAG chatbot",
)


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/cd224n-rag-agent")
async def ask_cs224n_agent(query: CS224NQueryInput) -> CS224NQueryOutput:
    try:
        query_response = await CS224N_rag_agent_executor.ainvoke({"input": query.text})
        query_response["intermediate_steps"] = [
            str(s) for s in query_response["intermediate_steps"]
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    return query_response


