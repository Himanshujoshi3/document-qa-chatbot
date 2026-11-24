"""Retrieval QA chain: history-aware retriever + Gemini answer generation."""
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import CHAT_MODEL, LLM_TEMPERATURE, RETRIEVER_K

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question which might "
            "reference context in the chat history, formulate a standalone "
            "question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and "
            "otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the user's documents.

Rules:
- Use ONLY the retrieved context below to answer. If the answer is not in
  the context, say you don't know — do not invent facts.
- Answer in the same language as the question: English, Hindi, or Hinglish
  (Hindi written in Roman letters).
- Keep answers concise and factual.

Context:
{context}"""


def get_llm(api_key=None, temperature=LLM_TEMPERATURE):
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL, google_api_key=api_key, temperature=temperature
    )


def build_qa_chain(vector_store, api_key=None):
    """Build a conversational retrieval chain over the vector store."""
    llm = get_llm(api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
