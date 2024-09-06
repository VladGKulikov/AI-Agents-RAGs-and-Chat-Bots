import os
from fastapi import FastAPI, HTTPException
from agent.agent import SelfRAGAgent, SelfRAGConfig
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class QueryInput(BaseModel):
    text: str


class QueryOutput(BaseModel):
    input: str
    output: str
    # intermediate_steps: list[str]

config = SelfRAGConfig(
    model=os.getenv('MODEL'),
    embed_model=os.getenv('EMBED_MODEL'),
    chroma_db_path=os.getenv('CHROMA_DB_PATH')
)

agent = SelfRAGAgent(config)


app = FastAPI(
    title="CS224N Stanford 'NLP and DL' course Chatbot",
    description="Endpoints for CS224N RAG chatbot",
)



@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/self-rag-agent")
async def ask_self_rag_agent(query: QueryInput) -> QueryOutput:
    try:
        print(f'!!!!query.text!!! = {query.text}. Type query = {type(query.text)}')
        query_response = await agent.graph.ainvoke({"question": query.text}) # query.text |
        print(f'!!!!query_response!!! = \n{query_response}\n')

        return QueryOutput(
            input=query.text,
            output=query_response['generation'],
            # intermediate_steps=[str(step) for step in query_response["intermediate_steps"]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))