
import pytest
import asyncio
from bot import AsyncChatBotWithMemory
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

@pytest.mark.asyncio
async def test_asynchronous_calls():
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

    async def async_call(session_id, question):
        return await bot.chat(session_id, question)

    session_ids = [bot.create_session() for _ in range(3)]
    tasks = [async_call(session_id, questions[i]) for i, session_id in enumerate(session_ids)]
    responses = await asyncio.gather(*tasks)

    for session_id, response, question in zip(session_ids, responses, questions):
        assert response is not None
        assert question in bot.sessions[session_id]

    assert len(session_ids) == 3
    print("Asynchronous test with multiple questions passed.")

if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(test_asynchronous_calls())
    end_time = time.perf_counter()
    print(f"Run time: {end_time - start_time} seconds")