"""Load PDFs and split them into chunks for embedding."""
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdfs(uploaded_files):
    """Load text from uploaded PDF file objects.

    Returns a list of Documents, one per page, with `source` (file name)
    and `page` (0-based) in each document's metadata.
    """
    docs = []
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            for page in PyPDFLoader(tmp_path).load():
                page.metadata["source"] = uploaded.name
                docs.append(page)
        finally:
            os.unlink(tmp_path)
    return docs


def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)
