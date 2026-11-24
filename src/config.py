"""Model and pipeline settings. One place to change models or chunk sizes."""

CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4
LLM_TEMPERATURE = 0.2

FAISS_INDEX_DIR = "faiss_index"
