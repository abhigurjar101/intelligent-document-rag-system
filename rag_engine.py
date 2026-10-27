"""
Intelligent Document Question-Answering System (RAG)
Orchestrated by Abhi Gurjar.
Parses heterogeneous legal documents, extracts semantic chunks, indexes vectors,
and answers complex queries with cited grounded evidence.
"""

import os
import re
import math
from typing import List, Dict, Any


def chunk_document(text: str, chunk_size: int = 250) -> List[str]:
    """Chunks text preserving numbered section headers and paragraphs."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    sections = re.split(r'\n(?=[0-9]+\.\s+)', text.strip())
    chunks = []
    for s in sections:
        cleaned = s.strip()
        if cleaned:
            words = cleaned.split()
            chunks.extend(" ".join(words[i:i + chunk_size])
                          for i in range(0, len(words), chunk_size))
    return chunks


def compute_tf_vector(text: str, vocab: Dict[str, int]) -> List[float]:
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    vec = [0.0] * len(vocab)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1.0
    norm = math.sqrt(sum(x ** 2 for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_sim(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


class DocumentRAGEngine:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.vocab = {}
        self.vectors = []

    def ingest_file(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        filename = os.path.basename(filepath)
        raw_chunks = chunk_document(content)
        
        for idx, chunk_text in enumerate(raw_chunks):
            self.chunks.append({
                "id": f"{filename}#clause_{idx+1}",
                "source": filename,
                "text": chunk_text
            })

        # Build vocabulary across all chunks
        all_words = set(re.findall(
            r'\b[a-zA-Z]{3,}\b', " ".join(c["text"] for c in self.chunks).lower()))
        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        self.vectors = [compute_tf_vector(c["text"], self.vocab) for c in self.chunks]
        print(f"Indexed {len(self.chunks)} clauses from '{filename}'.")

    def query(self, question: str, top_k: int = 2) -> Dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self.chunks:
            return {"error": "Knowledge base is empty."}

        q_vec = compute_tf_vector(question, self.vocab)
        scored = []
        for c, v in zip(self.chunks, self.vectors):
            sim = cosine_sim(q_vec, v)
            scored.append((sim, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = [(score, chunk) for score, chunk in scored if score > 0][:top_k]
        if not top_matches:
            return {"question": question, "citations": [], "relevance_score": 0.0,
                    "answer": "No matching evidence was found in the indexed documents.",
                    "retrieved_context": []}

        context_blocks = []
        citations = []
        for sim, item in top_matches:
            citations.append(item["id"])
            context_blocks.append(f"[{item['id']}]: {item['text']}")

        # Factual answer synthesis based on top matching clause
        best_text = top_matches[0][1]["text"] if top_matches else ""
        answer = f"Retrieved excerpt ({top_matches[0][1]['id']}): {best_text.splitlines()[0]} - {best_text}"

        return {
            "question": question,
            "citations": citations,
            "relevance_score": round(top_matches[0][0], 4) if top_matches else 0.0,
            "answer": answer,
            "retrieved_context": context_blocks
        }


def main():
    print("=" * 80)
    print("📜 INTELLIGENT DOCUMENT QUESTION-ANSWERING (RAG) SYSTEM")
    print("   Engineered by Abhi Gurjar")
    print("=" * 80)

    engine = DocumentRAGEngine()
    sample_path = os.path.join(os.path.dirname(__file__), "sample_contracts", "nda_agreement.txt")
    engine.ingest_file(sample_path)

    test_questions = [
        "How long do confidentiality obligations survive after the contract ends?",
        "What is the governing law and how are disputes resolved?",
        "What qualifies as an exclusion from confidential information?"
    ]

    for q in test_questions:
        res = engine.query(q, top_k=1)
        print(f"\n❓ Question: {res['question']}")
        print(f"   • Citation:  {res['citations'][0]}")
        print(f"   • Confidence Score: {res['relevance_score']}")
        print(f"   • Grounded Answer:\n     {res['answer']}")

    print("\n" + "=" * 80)
    print("Demo complete. Run the test suite for assertions.")
    print("=" * 80)


if __name__ == "__main__":
    main()
