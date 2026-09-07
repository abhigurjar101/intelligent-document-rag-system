# Intelligent Document Question-Answering System (RAG) 📜🤖

> **An enterprise Retrieval-Augmented Generation (RAG) system for extracting verified answers and clause citations from complex legal contracts and regulatory documents by Abhi Gurjar.**

---

## 💡 What Is This Project? (Explained Like a Human)

In corporate legal departments, banks, and procurement teams, attorneys and paralegals spend hundreds of hours reviewing non-disclosure agreements (NDAs), master service agreements (MSAs), and vendor contracts.

Finding answers to basic compliance questions—such as *"Which state's law governs this dispute?"* or *"How long do confidentiality covenants survive after termination?"*—traditionally required manual page-by-page scanning. If a key clause was overlooked, the company could face millions in liability.

**This project implements an Intelligent Document RAG System.** 
- It ingests unstructured legal agreements, extracts structured clause sections without breaking thoughts in half, and indexes them into high-dimensional vector representations.
- When an attorney asks a plain English question, the engine retrieves the exact contractual clause and synthesizes a direct, cited answer.
- **Zero hallucinations**: Every statement cites the exact contract file and clause number.

---

## 🏛️ Pipeline Architecture

```
[ Raw Legal PDF / TXT Agreements ]
               │
               ▼
   [ Clause & Section Chunker ]
               │
               ▼
    [ Vector Index & FAISS ]
               │
   ┌───────────┴───────────┐
   ▼                       ▼
[ Question Input ] ──► [ Semantic Search ] ──► [ Grounded Cited Answer ]
```

---

## 🚀 How to Run in 5 Seconds

Zero external API keys or paid cloud setups required to test the local engine:

```bash
# 1. Clone the repository
git clone https://github.com/abhigurjar101/intelligent-document-rag-system.git
cd intelligent-document-rag-system

# 2. Run the interactive RAG question-answering runner
python3 rag_engine.py
```

### What You Will See:
- Rapid ingestion and section indexing of `sample_contracts/nda_agreement.txt`.
- Semantic search across 3 complex legal questions (survival periods, governing law, confidentiality exclusions).
- Exact clause-by-clause citation output with similarity confidence scores.

---

## 📁 Repository Structure

```
intelligent-document-rag-system/
├── notebooks/
│   └── Extracting_Information_from_Legal_Documents_Using_RAG.ipynb  # End-to-end RAG development notebook
├── sample_contracts/
│   └── nda_agreement.txt                                            # Standard enterprise NDA test agreement
├── rag_engine.py                                                    # Standalone document RAG engine
├── requirements.txt                                                 # Dependencies (LangChain, FAISS, Sentence Transformers)
└── README.md                                                        # You are here! Simple human documentation
```

---

## 👤 Author

**Abhi Gurjar**  
- Portfolio: [abhigurjar.vercel.app](https://abhigurjar.vercel.app)  
- GitHub: [@abhigurjar101](https://github.com/abhigurjar101)  
- LinkedIn: [Abhi Gurjar](https://in.linkedin.com/in/abhi-gurjar-b13067203)
