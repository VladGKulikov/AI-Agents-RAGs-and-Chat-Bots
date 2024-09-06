1. add .env file to /backend 
OPENAI_API_KEY = 'your_OPENAI_API_KEY'  
LANGCHAIN_API_KEY = ''  
USER_AGENT = "DefaultLangchainUserAgent"  
MODEL = 'for example gpt-4o-mini'  
EMBED_MODEL = 'for example text-embedding-3-small'  
CHROMA_DB_PATH = 'for example vector-db'  

2. add your chroma DB: /backend/vector-db 

2. docker-compose up --build

