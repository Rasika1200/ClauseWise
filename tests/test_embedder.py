from collections import namedtuple

import pytest

from modules import embedder

Chunk = namedtuple("Chunk", ["text", "chunk_id", "source_file", "page_num"])

CHUNKS = [
    Chunk("This Agreement is governed by the laws of the State of New York.", "c1", "test.pdf", 1),
    Chunk("Neither party may assign this Agreement without prior written consent.", "c2", "test.pdf", 2),
]


@pytest.fixture(autouse=True)
def tfidf_only(monkeypatch):
    # Keep the test fast and offline: TF-IDF embeddings, fresh in-memory collection
    monkeypatch.setenv("CLAUSEWISE_NEURAL", "0")
    embedder.clear_collection()
    yield
    embedder.clear_collection()


def _stored_metadata():
    got = embedder.get_collection().get(include=["metadatas"])
    return {m["chunk_id_original"]: m for m in got["metadatas"]}


def test_index_chunks_accepts_match_chunks_labels():
    # Shape produced by clause_matcher.match_chunks, which the app and API pass in
    labels = [
        {"matched_clause": "governing law", "match_score": 0.71, "risk_level": "low"},
        {"matched_clause": "assignment", "match_score": 0.64, "risk_level": "medium"},
    ]
    embedder.index_chunks(CHUNKS, topic_labels=labels)

    meta = _stored_metadata()
    assert meta["c1"]["clause_type"] == "governing law"
    assert meta["c1"]["confidence"] == "0.71"
    assert meta["c2"]["clause_type"] == "assignment"
    assert meta["c2"]["risk_level"] == "medium"

    hits = embedder.semantic_search("transfer the contract", top_k=5, filter_clause_type="assignment")
    assert [h["metadata"]["chunk_id_original"] for h in hits] == ["c2"]


def test_index_chunks_accepts_topic_modeler_labels():
    labels = [
        {"clause_type": "governing law", "confidence": 0.9, "risk_level": "low", "summary": "NY law"},
        {"clause_type": "assignment", "confidence": 0.8, "risk_level": "medium", "summary": ""},
    ]
    embedder.index_chunks(CHUNKS, topic_labels=labels)

    meta = _stored_metadata()
    assert meta["c1"]["clause_type"] == "governing law"
    assert meta["c1"]["confidence"] == "0.9"
    assert meta["c1"]["summary"] == "NY law"


def test_index_chunks_without_labels_has_no_clause_type():
    embedder.index_chunks(CHUNKS, topic_labels=None)
    assert "clause_type" not in _stored_metadata()["c1"]
