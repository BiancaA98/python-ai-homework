from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from homework_03 import tema3


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        # Vectori controlati: testam Chroma real fara descarcarea modelelor.
        vectors = {"relevant": [1.0, 0.0], "boundary": [0.5, 0.8660254],
                   "unrelated": [0.0, 1.0], "opposite": [-1.0, 0.0]}
        return np.array([vectors.get(text, [1.0, 0.0]) for text in texts])


@pytest.fixture
def collection(tmp_path):
    return tema3.open_collection(tmp_path / "db")


def test_ingestion_persistence_no_duplicates_and_updates(collection, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "notes.txt"
    source.write_text("A" * 1800, encoding="utf-8")
    embedder = FakeEmbedder()
    tema3.ingest_documents(collection, data, embedder)
    first = collection.get(include=["metadatas"])
    tema3.ingest_documents(collection, data, embedder)
    assert collection.count() == len(first["ids"]) == 3
    assert collection.get()["ids"] == first["ids"]
    assert all(meta == {"source": "notes.txt"} for meta in first["metadatas"])
    reopened = tema3.open_collection(tmp_path / "db")
    assert reopened.count() == 3
    source.write_text("Changed text", encoding="utf-8")
    tema3.ingest_documents(reopened, data, embedder)
    assert reopened.count() == 1
    assert reopened.get()["documents"] == ["Changed text"]


def test_pdf_pages_and_empty_page(collection, tmp_path, monkeypatch):
    (tmp_path / "notes.pdf").touch()
    reader = SimpleNamespace(pages=[
        SimpleNamespace(extract_text=lambda: "First page"),
        SimpleNamespace(extract_text=lambda: None),
        SimpleNamespace(extract_text=lambda: "Third page"),
    ])
    monkeypatch.setattr(tema3, "PdfReader", lambda path: reader)
    tema3.ingest_documents(collection, tmp_path, FakeEmbedder())
    result = collection.get()
    assert sorted(meta["page"] for meta in result["metadatas"]) == [1, 3]
    assert all(meta["source"] == "notes.pdf" for meta in result["metadatas"])


def test_load_real_pdf(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 10 100 Td (PDF test text) Tj ET")
    page[NameObject("/Contents")] = stream
    path = tmp_path / "fixture.pdf"
    writer.write(path)
    pages = tema3.load_document(path)
    assert pages[0][1] == 1
    assert "PDF test text" in pages[0][0]


def test_filename_filter_and_distance_boundary(collection):
    collection.upsert(
        ids=["a", "b", "c", "other"],
        documents=["relevant", "boundary", "unrelated", "other file"],
        embeddings=[[1.0, 0.0], [0.5, 0.8660254], [0.0, 1.0], [1.0, 0.0]],
        metadatas=[{"source": "notes.txt"}] * 3 + [{"source": "other.txt"}],
    )
    results = tema3.search(collection, "notes.txt", "relevant", embedder=FakeEmbedder())
    assert [item["document"] for item in results] == ["relevant", "boundary"]
    assert all(item["metadata"]["source"] == "notes.txt" for item in results)
    assert all(item["distance"] <= 0.5 for item in results)
    strict = tema3.search(collection, "notes.txt", "relevant", 0.1, FakeEmbedder())
    assert [item["document"] for item in strict] == ["relevant"]
    embedder = Mock()
    with pytest.raises(ValueError, match="File not indexed"):
        tema3.search(collection, "missing.txt", "question", embedder=embedder)
    embedder.encode.assert_not_called()


def test_search_limits_candidates_to_15(collection):
    collection.upsert(ids=[str(i) for i in range(20)],
                      documents=["relevant"] * 20, embeddings=[[1.0, 0.0]] * 20,
                      metadatas=[{"source": "notes.txt"}] * 20)
    assert len(tema3.search(collection, "notes.txt", "relevant",
                           embedder=FakeEmbedder())) == 15


def test_rerank_threshold_order_and_top_four():
    chunks = [{"document": str(i)} for i in range(7)]
    reranker = Mock()
    reranker.predict.return_value = [0.2, 0.5, 0.6, 0.9, 0.7, 0.8, 0.1]
    assert tema3.rerank("question", chunks, 0.5, reranker) == [chunks[i] for i in [3, 5, 4, 2]]
    assert tema3.rerank("question", chunks, 0.95, reranker) == []
    reranker.predict.assert_called_with([("question", str(i)) for i in range(7)])


@pytest.mark.parametrize("use_rerank", [False, True])
def test_no_results_skips_llm_and_reranker(monkeypatch, use_rerank):
    monkeypatch.setattr(tema3, "search", Mock(return_value=[]))
    llm = Mock()
    model = Mock()
    monkeypatch.setattr(tema3, "generate_answer", llm)
    monkeypatch.setattr(tema3, "get_reranker", model)
    assert tema3.answer_question(Mock(), "notes.txt", "question",
                                use_rerank=use_rerank) == tema3.NO_RESULTS
    llm.assert_not_called()
    model.assert_not_called()


def test_reranking_rejects_all_without_llm(monkeypatch):
    monkeypatch.setattr(tema3, "search", Mock(return_value=[{"document": "irrelevant"}]))
    model = Mock()
    model.predict.return_value = [0.1]
    monkeypatch.setattr(tema3, "get_reranker", lambda: model)
    llm = Mock()
    monkeypatch.setattr(tema3, "generate_answer", llm)
    assert tema3.answer_question(Mock(), "notes.txt", "question", use_rerank=True) == tema3.NO_RESULTS
    llm.assert_not_called()


def test_answer_uses_only_top_four_reranked_chunks(monkeypatch):
    chunks = [{"document": str(i)} for i in range(6)]
    monkeypatch.setattr(tema3, "search", Mock(return_value=chunks))
    model = Mock()
    model.predict.return_value = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    monkeypatch.setattr(tema3, "get_reranker", lambda: model)
    llm = Mock(return_value="answer")
    monkeypatch.setattr(tema3, "generate_answer", llm)
    assert tema3.answer_question(Mock(), "notes.txt", "question", use_rerank=True) == "answer"
    llm.assert_called_once_with("question", [chunks[i] for i in [5, 4, 3, 2]])


def test_llm_context_and_source_references(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPEN_ROUTER_MODEL_NAME", "test-model")
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))])
    monkeypatch.setattr(tema3, "OpenAI", lambda **kwargs: client)
    chunks = [
        {"document": "PDF context", "metadata": {"source": "notes.pdf", "page": 2}},
        {"document": "Text context", "metadata": {"source": "notes.txt"}},
    ]
    answer = tema3.generate_answer("question", chunks)
    assert "notes.pdf, pagina 2" in answer
    assert "notes.txt" in answer
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert "Nu inventa" in messages[0]["content"]
    assert "PDF context" in messages[1]["content"]
    assert "Text context" in messages[1]["content"]


def test_empty_data_directory_needs_no_model(collection, tmp_path, monkeypatch, capsys):
    model = Mock()
    monkeypatch.setattr(tema3, "get_embedder", model)
    assert tema3.ingest_documents(collection, tmp_path) == 0
    assert "Add TXT, MD or PDF files" in capsys.readouterr().out
    model.assert_not_called()


def test_chunk_overlap_and_validation():
    assert tema3.chunk_text("abcdefghij", size=6, overlap=2) == ["abcdef", "efghij"]
    assert tema3.chunk_text("   ") == []
    with pytest.raises(ValueError):
        tema3.chunk_text("text", size=2, overlap=2)
