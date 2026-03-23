# Local PDF RAG System

A lightweight, local Retrieval-Augmented Generation (RAG) system that allows you to chat with your PDF documents. This project uses Ollama for local embeddings and LLM generation, FAISS for efficient similarity search, and Python for the orchestration.

## 🚀 Features

- **100% Local:** No data leaves your machine. Privacy-focused using Ollama.
- **Efficient Vector Search:** Uses Facebook AI Similarity Search (FAISS) for lightning-fast document retrieval.
- **Smart Chunking:** Automatically breaks down large PDFs into manageable chunks with estimated page tracking.
- **Persistence:** Saves processed vectors and text chunks locally so you don't have to re-process the same PDF twice.

## 🏗️ Architecture
```
PDF → Chunking → Embeddings (nomic-embed-text) → FAISS Index
Query → Embed → FAISS Search → Top-3 Chunks → Mistral → Answer
```

## 📊 Evaluation (RAGAS Metrics)

Evaluated using the [RAGAS](https://github.com/explodinggradients/ragas) framework on a sample document QA set.

| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Context Recall | 1.00 |
| Context Precision | 0.87 |
| Answer Relevancy | 0.77 |

**Key findings:**
- Perfect faithfulness (1.0) — model answers strictly from retrieved context, no hallucination
- Perfect context recall (1.0) — all relevant chunks are being retrieved
- Answer relevancy (0.77) — identified response verbosity as root cause, improved via prompt optimization

## 🛠️ Tech Stack

- **LLM & Embeddings:** Ollama (mistral + nomic-embed-text)
- **Vector Database:** FAISS
- **PDF Processing:** PyPDF2
- **Evaluation:** RAGAS
- **Language:** Python 3.x

## 📋 Prerequisites

Install Ollama from [ollama.com](https://ollama.com), then pull required models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

## 📂 Project Structure

| File | Description |
|---|---|
| `main.py` | Central entry point |
| `pdf_to_vector.py` | PDF extraction, chunking, embedding, FAISS index creation |
| `question_vector.py` | Similarity search and answer generation |
| `vectors.index` | (Generated) FAISS vector database |
| `chunks.pkl` | (Generated) Pickled text data and metadata |

## 🚀 Usage
```bash
python main.py
```

- **Option 1** — Process a new PDF: enter path like `data/my_doc.pdf`
- **Option 2** — Ask questions: retrieves top-3 chunks and generates answer

## 🧠 How it Works

1. **Ingestion** — PDF split into ~500 character chunks
2. **Embedding** — Each chunk vectorized using `nomic-embed-text`
3. **Indexing** — Vectors stored in FAISS using Inner Product similarity
4. **Retrieval** — Query embedded → FAISS finds top-3 matching chunks
5. **Generation** — Chunks passed as context to Mistral for answer generation