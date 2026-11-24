"""Pipeline tests — run offline, no Gemini key needed.

They use a tiny deterministic fake embedding so the FAISS round-trip and
the chunking logic can be verified without any API call:

    pip install -r requirements-dev.txt
    pytest -q
"""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.pdf_loader import load_pdfs, split_documents
from src.vector_store import build_vector_store  # noqa: F401  (import check)


class FakeEmbeddings(Embeddings):
    """Tiny keyword-based embeddings — test-only.

    Dimension 0 counts "cat", dimension 1 counts "dog", so a "cat"
    query ranks the cat document first. Deterministic, no API call.
    """

    def _embed(self, text):
        t = text.lower()
        return [float(t.count("cat")), float(t.count("dog")), 0.0, 0.0]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_split_documents_overlaps_chunks():
    docs = [Document(page_content="word " * 500, metadata={"source": "a.pdf"})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 100 for c in chunks)


def test_faiss_retrieves_matching_chunk(tmp_path):
    docs = [
        Document(page_content="The cat sits on the mat.", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="Dogs love long walks in the park.", metadata={"source": "b.pdf", "page": 1}),
    ]
    store = FAISS.from_documents(docs, FakeEmbeddings())
    store.save_local(str(tmp_path / "index"))
    loaded = FAISS.load_local(
        str(tmp_path / "index"), FakeEmbeddings(), allow_dangerous_deserialization=True
    )
    results = loaded.similarity_search("cat", k=1)
    assert "cat" in results[0].page_content
    assert results[0].metadata["source"] == "a.pdf"


def test_load_sample_pdf():
    """Loads the bundled sample PDF — proves the loader works end to end."""

    class FakeUpload:
        name = "sample.pdf"

        def __init__(self, path):
            self._bytes = open(path, "rb").read()

        def read(self):
            return self._bytes

    docs = load_pdfs([FakeUpload("sample_docs/sample.pdf")])
    assert len(docs) >= 1
    assert docs[0].metadata["source"] == "sample.pdf"
    assert "leave" in docs[0].page_content.lower()
