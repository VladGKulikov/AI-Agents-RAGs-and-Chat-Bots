import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import uuid
from typing import Dict

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
MODEL = 'gpt-4o-mini'  # 128K context

class AsyncChatBotWithMemory:
    def __init__(self, system_prompt="You are a helpful assistant."):
        self.sessions: Dict[str, list] = {}  # Dictionary for storing sessions
        self.system_prompt = system_prompt  # Persistent system prompt         

    def create_session(self):
        """Creates a new session with a unique identifier"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id

    def remember(self, session_id, message):
        """Adds a message to the memory of a specific session"""
        if session_id in self.sessions:
            self.sessions[session_id].append(message)
        else:
            raise ValueError("Session not found.")

    def forget(self, session_id):
        """Clears the memory of a specific session"""
        if session_id in self.sessions:
            self.sessions[session_id] = []
        else:
            raise ValueError("Session not found.")

    async def chat(self, session_id, message):
        """Asynchronously sends a message to OpenAI and receives a response, taking memory into account."""
        if session_id not in self.sessions:
            raise ValueError("Session not found.")

        self.remember(session_id, message)
        
        # Forming the context for an API request, including previous messages
        context = "\n".join(self.sessions[session_id])
        
        response = await async_client.chat.completions.create(
            model=MODEL,              
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]
        )
        
        # Extracting the response text
        answer = response.choices[0].message.content        
        self.remember(session_id, answer)
        
        return answer