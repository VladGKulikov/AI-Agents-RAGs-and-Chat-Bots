# If not in main type in CLI: uvicorn main:app --reload
# Go to: http://127.0.0.1:8000/static/index.html

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from bot import AsyncChatBotWithMemory

app = FastAPI()

# Creating an instance of the bot with a custom system prompt
system_prompt = "You are a friendly and helpful assistant. Always be polite and provide detailed responses."
bot = AsyncChatBotWithMemory(system_prompt=system_prompt)

# Connecting a folder for static files.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/create_session")
async def create_session():
    """API endpoint to create a new session and return the session ID"""
    session_id = bot.create_session()
    return {"session_id": session_id}

@app.post("/chat/{session_id}")
async def chat(session_id: str, message: str):
    """API endpoint to send a message to the bot and get a response"""
    response = await bot.chat(session_id, message)
    return {"response": response}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint to handle real-time communication with the bot"""
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = await bot.chat(session_id, data)
        await websocket.send_text(response)

@app.get("/")
async def get():
    """API endpoint to serve the main HTML page"""
    return FileResponse("static/index.html")



if __name__ == "__main__":
    # Start the Uvicorn server    
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)