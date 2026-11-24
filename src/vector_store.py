"""FAISS vector store backed by Gemini embeddings."""
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL, FAISS_INDEX_DIR


def get_embeddings(api_key=None):
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)


def build_vector_store(chunks, api_key=None, persist_dir=FAISS_INDEX_DIR):
    """Embed chunks and persist the FAISS index to disk."""
    store = FAISS.from_documents(chunks, get_embeddings(api_key))
    store.save_local(persist_dir)
    return store


def load_vector_store(api_key=None, persist_dir=FAISS_INDEX_DIR):
    """Load a previously saved FAISS index."""
    return FAISS.load_local(
        persist_dir, get_embeddings(api_key), allow_dangerous_deserialization=True
    )
