from modules.rag_pipeline import assemble_context, rerank, rag_retrieve_and_assemble

def test_assemble_context():
    chunks = [
        {
            "text": "The liability is capped at $50,000.",
            "metadata": {"source_file": "contract.pdf", "page_num": 3, "clause_type": "liability", "risk_level": "medium"}
        },
        {
            "text": "This contract is governed by California law.",
            "metadata": {"source_file": "contract.pdf", "page_num": 5, "clause_type": "governing law", "risk_level": "low"}
        }
    ]
    context = assemble_context(chunks)
    assert "[Chunk 1 | File: contract.pdf | Page: 3 | Clause: liability | Risk: medium]" in context
    assert "The liability is capped at $50,000." in context
    assert "[Chunk 2 | File: contract.pdf | Page: 5 | Clause: governing law | Risk: low]" in context
    assert "California law." in context

def test_rerank():
    candidates = [
        {"text": "low similarity", "similarity": 0.3},
        {"text": "high similarity", "similarity": 0.8},
        {"text": "medium similarity", "similarity": 0.5}
    ]
    
    # Reranking without a cross-encoder should sort by similarity descending
    reranked = rerank("test query", candidates)
    assert reranked[0]["text"] == "high similarity"
    assert reranked[1]["text"] == "medium similarity"
    assert reranked[2]["text"] == "low similarity"

def test_rag_retrieve_and_assemble(mocker):
    mock_retrieved = [
        {
            "text": "California law governs.",
            "metadata": {"source_file": "test.pdf", "page_num": 1, "clause_type": "governing law"}
        }
    ]
    # Mock retrieve function to return mock list
    mocker.patch("modules.rag_pipeline.retrieve", return_value=mock_retrieved)
    
    context, chunks = rag_retrieve_and_assemble("what is the governing law?")
    assert len(chunks) == 1
    assert "California law governs." in context
    assert chunks[0]["text"] == "California law governs."
