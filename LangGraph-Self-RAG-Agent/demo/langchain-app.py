import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Настройка клиента ChromaDB
chroma_client = Chroma(
    host=os.environ.get("CHROMA_SERVER_HOST", "localhost"),
    port=int(os.environ.get("CHROMA_SERVER_HTTP_PORT", 8000))
)

# Инициализация embeddings
EMBED_MODEL = os.getenv('EMBED_MODEL', 'text-embedding-ada-002')
embedding = OpenAIEmbeddings(model=EMBED_MODEL)

# Подключение к существующей коллекции
vector_store = Chroma(
    client=chroma_client,
    embedding_function=embedding
)

# Пример использования: поиск в существующей базе данных
query = "What are transformers in natural language processing?"
docs = vector_store.similarity_search(query, k=2)
print(f"Результаты поиска для запроса '{query}':")
for doc in docs:
    print(f"- {doc.page_content[:200]}...")

# Пример добавления новых данных (если требуется)
# new_texts = ["Transformers have revolutionized NLP tasks", "BERT is a popular transformer model"]
# vector_store.add_texts(new_texts)
# print("\nДобавлены новые данные в базу.")
