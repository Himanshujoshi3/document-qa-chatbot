# Document Q&A Chatbot

Ask questions about your PDF documents — in English, Hindi, or Hinglish — and get
answers grounded in the documents, with source citations.

## How it works

```
PDF upload → text extraction → chunks (1000 chars, 200 overlap)
           → Gemini embeddings → FAISS vector store
           → history-aware retriever → Gemini 2.5 Flash → answer + sources
```

- **RAG, not fine-tuning:** the model only answers from your uploaded documents.
  If the answer isn't in them, it says so instead of inventing one.
- **Conversation memory:** follow-up questions work — "and what about sick leave?"
  is understood in context.
- **Source citations:** every answer lists the file and page it came from.
- **Hinglish:** ask in Hindi written in Roman letters, get an answer back the same way.

## Tech stack

Python · Streamlit · LangChain · FAISS · Gemini API (embeddings + LLM)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your Gemini API key in .env
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## Run

```bash
streamlit run app.py
```

1. Paste your Gemini API key in the sidebar (or set it in `.env`).
2. Upload one or more PDFs.
3. Click **Process documents**.
4. Ask questions in the chat.

Try it instantly with the bundled `sample_docs/sample.pdf` — ask
"How many casual leaves do employees get?" (answer: 18).

## Tests

No API key needed — the tests use a deterministic fake embedding:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Project structure

```
app.py                  # Streamlit UI + chat loop
src/
  config.py             # models, chunk size, retriever settings
  pdf_loader.py         # PDF loading + chunking
  vector_store.py       # Gemini embeddings + FAISS build/load
  qa_engine.py          # history-aware retrieval QA chain
tests/
  test_pipeline.py      # offline pipeline tests (fake embeddings)
sample_docs/
  sample.pdf            # demo document to try the app with
```
