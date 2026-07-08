"""
Module 1: Document Processor
Extracts text from PDF/DOCX contracts, cleans it, and chunks it into
overlapping segments suitable for embedding and topic modelling.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from docx import Document


@dataclass
class Chunk:
    text: str
    chunk_id: str
    source_file: str
    page_num: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


def extract_text_from_pdf(file_path: str) -> List[dict]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    pages = []
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text = _clean_text(text)
        if text.strip():
            pages.append({"page_num": page_num + 1, "text": text})
    doc.close()
    return pages


def extract_text_from_docx(file_path: str) -> List[dict]:
    """Extract text paragraph-by-paragraph from a DOCX file."""
    doc = Document(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    full_text = _clean_text(full_text)
    # Treat entire doc as page 1 for simplicity
    return [{"page_num": 1, "text": full_text}]


def extract_text(file_path: str) -> List[dict]:
    """Auto-detect file type and extract text."""
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(file_path)
    elif path.suffix.lower() in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and fix common PDF extraction artifacts."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    return text.strip()


class SemanticChunk(Chunk):
    """Subclass of Chunk that supports dictionary-like indexing for Phase 2 spec compatibility."""
    def __getitem__(self, key):
        if key == "text":
            return self.text
        elif key == "source":
            return self.source_file
        elif key == "chunk_index":
            return self.metadata.get("chunk_index")
        elif key == "strategy":
            return self.metadata.get("strategy")
        elif key == "word_count":
            return self.metadata.get("word_count")
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return ["text", "source", "chunk_index", "strategy", "word_count"]

    def items(self):
        return [(k, self[k]) for k in self.keys()]


def semantic_chunk_text(text: str, source_file: str) -> List[Chunk]:
    """
    Chunk text semantically using sentence embeddings.
    1. Splits text into sentences.
    2. Batch embeds all sentences using sentence-transformers.
    3. Detects boundaries where cosine similarity drops below 0.65.
    4. Merges small blocks (< 60 words) and splits large blocks (> 400 words).
    """
    import numpy as np
    from modules.embedder import embed_texts

    # 1. Split text into sentences (split on ., ?, ! followed by whitespace or end of string)
    raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    if not sentences:
        return []

    # 2. Batch generate embeddings for sentences
    embeddings = embed_texts(sentences)
    if not embeddings:
        return []

    # 3. Compute cosine similarity between consecutive sentence pairs
    vecs = np.array(embeddings)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_vecs = vecs / norms

    similarities = []
    for i in range(len(normalized_vecs) - 1):
        sim = np.dot(normalized_vecs[i], normalized_vecs[i+1])
        similarities.append(float(sim))

    # 4. Detect boundaries where similarity drops below 0.65
    boundaries = []
    for idx, sim in enumerate(similarities):
        if sim < 0.65:
            boundaries.append(idx)

    # 5. Group sentences into blocks
    blocks = []
    start_idx = 0
    for b_idx in boundaries:
        blocks.append(sentences[start_idx:b_idx + 1])
        start_idx = b_idx + 1
    if start_idx < len(sentences):
        blocks.append(sentences[start_idx:])

    # 6. Merge blocks with fewer than 60 words into adjacent block (prefer next; merge with previous if at end)
    i = 0
    while i < len(blocks):
        curr_block = blocks[i]
        curr_words = sum(len(s.split()) for s in curr_block)
        if curr_words < 60:
            if i + 1 < len(blocks):
                # Merge with next
                blocks[i] = curr_block + blocks[i+1]
                blocks.pop(i+1)
            elif i - 1 >= 0:
                # Merge with previous
                blocks[i-1] = blocks[i-1] + curr_block
                blocks.pop(i)
            else:
                # Only one block remains, cannot merge
                i += 1
        else:
            i += 1

    # 7. Split blocks exceeding 400 words at the nearest sentence boundary at or before 400-word mark
    final_blocks = []
    for block in blocks:
        if sum(len(s.split()) for s in block) <= 400:
            final_blocks.append(block)
            continue

        temp_block = list(block)
        while sum(len(s.split()) for s in temp_block) > 400:
            cum_words = 0
            split_idx = -1
            for s_idx, s in enumerate(temp_block):
                s_words = len(s.split())
                if cum_words + s_words > 400:
                    break
                cum_words += s_words
                split_idx = s_idx
            
            if split_idx == -1:
                split_idx = 0  # Force split at first sentence if single sentence is > 400 words

            final_blocks.append(temp_block[:split_idx + 1])
            temp_block = temp_block[split_idx + 1:]

        if temp_block:
            final_blocks.append(temp_block)

    # 8. Return as SemanticChunk list
    ret_chunks = []
    for idx, b in enumerate(final_blocks):
        block_text = " ".join(b)
        n_words = sum(len(s.split()) for s in b)
        
        chunk = SemanticChunk(
            text=block_text,
            chunk_id=f"{Path(source_file).stem}_sem_c{idx}",
            source_file=source_file,
            page_num=1,
            char_start=0,
            char_end=len(block_text),
            metadata={
                "chunk_index": idx,
                "strategy": "semantic",
                "word_count": n_words
            }
        )
        ret_chunks.append(chunk)

    return ret_chunks


def chunk_pages(
    pages: List[dict],
    source_file: str,
    chunk_size: int = 500,
    overlap: int = 80,
    strategy: str = "semantic",
) -> List[Chunk]:
    """
    Split pages into chunks using either the "semantic" sentence similarity 
    boundary drop strategy or the naive "sliding_window" strategy.
    """
    if strategy == "semantic":
        full_text = "\n".join(p["text"] for p in pages)
        return semantic_chunk_text(full_text, source_file)

    # Existing sliding-window logic
    chunks = []
    chunk_idx = 0

    for page in pages:
        words = page["text"].split()
        page_num = page["page_num"]
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{Path(source_file).stem}_p{page_num}_c{chunk_idx}",
                source_file=source_file,
                page_num=page_num,
                char_start=start,
                char_end=end,
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end == len(words):
                break
            start += chunk_size - overlap

    return chunks


def process_raw_text(
    text: str,
    source_file: str = "Webpage_Text",
    chunk_size: int = 500,
    overlap: int = 80,
    strategy: str = "semantic",
) -> List[Chunk]:
    """
    Process raw text strings directly, bypassing PDF/DOCX extraction.
    Used for extracting text from the Chrome Extension DOM.
    """
    cleaned_text = _clean_text(text)
    pages = [{"page_num": 1, "text": cleaned_text}]
    return chunk_pages(pages, source_file=source_file, chunk_size=chunk_size, overlap=overlap, strategy=strategy)


def process_contract(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 80,
    strategy: str = "semantic",
) -> Tuple[List[Chunk], str]:
    """
    End-to-end: extract text from file, compute SHA-256 hash of the text,
    and return chunks and doc_hash.
    Saves a contract analysis placeholder row to the database.
    """
    import hashlib
    from modules import db_manager

    pages = extract_text(file_path)
    full_text = "\n".join(p["text"] for p in pages)
    doc_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

    chunks = chunk_pages(pages, source_file=file_path, chunk_size=chunk_size, overlap=overlap, strategy=strategy)
    
    # Save database placeholder row
    try:
        db_manager.save_contract_analysis(
            contract_name=Path(file_path).name,
            contract_hash=doc_hash,
            risk_profile="",
            summary="",
            assessment=""
        )
    except Exception as e:
        print(f"Error saving database placeholder analysis: {e}")

    return chunks, doc_hash


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run user semantic chunking test
    print("Testing doc_processor.py semantic chunking...")
    legal_para = """
    This Agreement is entered into on this 30th day of June, 2026, by and between Company A (hereinafter "Discloser") and Company B (hereinafter "Recipient").
    The parties wish to explore a potential business relationship concerning AI technologies.
    For this purpose, Discloser may disclose certain confidential and proprietary information to Recipient.
    Recipient agrees to maintain the confidentiality of all such proprietary information.
    Recipient shall not disclose any confidential information to third parties without the prior written consent of Discloser.
    The obligations of confidentiality shall survive the termination of this Agreement for a period of five (5) years.
    Either party may terminate this Agreement at any time upon thirty (30) days written notice to the other party.
    Upon termination, Recipient shall return or destroy all copies of confidential information in its possession.
    This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflict of law principles.
    Any disputes arising out of or in connection with this Agreement shall be resolved through binding arbitration in Wilmington, Delaware.
    The prevailing party in any action to enforce this Agreement shall be entitled to recover its reasonable attorney's fees and court costs.
    In witness whereof, the parties have executed this Agreement as of the date first written above.
    """
    # Multiply to get over 300 words of legal text
    sample_text = (legal_para + "\n") * 3
    
    chunks_sem = semantic_chunk_text(sample_text, "test_contract.pdf")
    print(f"Semantic chunk count: {len(chunks_sem)}")
    assert 1 <= len(chunks_sem) <= 10, f"Expected chunk count between 1 and 10, got {len(chunks_sem)}!"
    
    for i, c in enumerate(chunks_sem):
        w_count = c["word_count"]
        print(f"  Chunk {i}: {w_count} words")
        assert 60 <= w_count <= 400, f"Expected chunk {i} to have word_count between 60 and 400, got {w_count}!"

    print("doc_processor semantic chunking verification passed")

