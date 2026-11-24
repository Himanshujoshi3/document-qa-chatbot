"""Document Q&A Chatbot — ask questions about your PDFs.

RAG pipeline: PDF -> chunks -> Gemini embeddings -> FAISS -> Gemini LLM.
Answers in the same language as the question: English, Hindi, or Hinglish.

Run:
    streamlit run app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.pdf_loader import load_pdfs, split_documents
from src.qa_engine import build_qa_chain
from src.vector_store import build_vector_store

load_dotenv()

st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📄", layout="wide")

st.title("📄 Document Q&A Chatbot")
st.caption("Upload PDFs, then ask questions — answers come from your documents, with sources.")

# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str, "sources": [...]}]
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ---------- sidebar ----------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Gemini API key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Free key from https://aistudio.google.com/apikey — or set GOOGLE_API_KEY in a .env file.",
    )
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        if not api_key:
            st.error("Add your Gemini API key first.")
        else:
            with st.spinner("Reading PDFs, chunking, embedding…"):
                docs = load_pdfs(uploaded_files)
                chunks = split_documents(docs)
                store = build_vector_store(chunks, api_key=api_key)
                st.session_state.qa_chain = build_qa_chain(store, api_key=api_key)
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.messages = []
            st.success(f"Ready — {len(docs)} pages → {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.processed_files:
        st.subheader("Indexed documents")
        for name in st.session_state.processed_files:
            st.write(f"• {name}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- chat ----------
if st.session_state.qa_chain is None:
    st.info("👆 Upload one or more PDFs in the sidebar and click **Process documents** to start.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.write(f"**{src['source']}** — page {src['page']}")
                        st.caption(src["text"])

    question = st.chat_input("Ask about your documents… (English, Hindi, or Hinglish)")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history = []
        for m in st.session_state.messages[:-1]:
            history.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                           else AIMessage(content=m["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = st.session_state.qa_chain.invoke(
                    {"input": question, "chat_history": history}
                )
            answer = result["answer"]
            st.markdown(answer)
            sources = [
                {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page", 0) + 1,
                    "text": d.page_content[:300],
                }
                for d in result.get("context", [])
            ]
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.write(f"**{src['source']}** — page {src['page']}")
                        st.caption(src["text"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
