# Document retrieval baseline

A small, inspectable Python baseline for document question answering. It splits UTF-8 text into numbered sections and bounded word chunks, builds normalized term-frequency vectors, ranks chunks by cosine similarity, and returns source excerpts with citations.

This repository provides a lexical retrieval baseline for the document-search work described in my CV. The command-line engine does not use an LLM, LangChain, FAISS, Qdrant, or neural embeddings. The notebook is a separate exploration; its optional dependencies are listed in `requirements.txt`.

## Run

```bash
git clone https://github.com/abhigurjar101/intelligent-document-rag-system.git
cd intelligent-document-rag-system
python3 rag_engine.py
python3 -m unittest discover -s tests -v
```

The CLI and tests use Python's standard library. The bundled NDA is demonstration text, not legal advice or a client document.

## How it works

`ingest_file(path)` reads text, chunks it, rebuilds the vocabulary across all indexed chunks, and recomputes vectors. `query(question, top_k=2)` returns excerpts with `filename#clause_N` identifiers. Queries with no token overlap return no evidence. Similarity is a retrieval score, not a probability of correctness.

## Limits and next experiments

Storage is in memory; PDF parsing, duplicate-ingestion handling, persistent indexes, access control, and a trained reranker are not implemented in the CLI. Lexical overlap can miss paraphrases and retrieve passages that share words but do not answer the question. Evaluate recall@k and citation accuracy on a labeled question set before comparing this baseline with a neural RAG system. No accuracy or business-impact benchmark is claimed.

## Structure

- `rag_engine.py`: baseline and CLI demonstration
- `sample_contracts/`: sample input
- `notebooks/`: separate legal-document exploration
- `tests/`: regression tests for multi-file retrieval, no-match behavior, chunk bounds and validation

[Abhi Kumar Gurjar](https://www.linkedin.com/in/abhigurjar) · Data Scientist & GenAI Engineer
