
import asyncio
from bot import AsyncChatBotWithMemory
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI


def test_synchronous_calls():
    bot = AsyncChatBotWithMemory()
    questions = [
        # "How far is the sun?"
        #"Tell me a joke.",
        "What is 2 + 2?",
        "Who is the president of the United States?",
        "What is the capital of France?" #,
        # "How do you make a cake?",
        #"What time is it?",
        #"Can you sing a song?",
        #"What is the largest animal on Earth?",
        #"How far is the moon?"
    ]

    session_ids = []
    for i in range(3):
        session_id = bot.create_session()
        session_ids.append(session_id)        
        response = asyncio.run(bot.chat(session_id, questions[i]))
        assert response is not None
        assert questions[i] in bot.sessions[session_id]

    assert len(session_ids) == 3
    print("Synchronous test with multiple questions passed.")

if __name__ == "__main__":
    start_time = time.perf_counter()
    test_synchronous_calls()
    end_time = time.perf_counter()
    print(f"Run time: {end_time - start_time} seconds")
