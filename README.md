Local PDF RAG System

A lightweight, local Retrieval-Augmented Generation (RAG) system that allows you to chat with your PDF documents. This project uses Ollama for local embeddings and LLM generation, FAISS for efficient similarity search, and Python for the orchestration.

🚀 Features
100% Local: No data leaves your machine. Privacy-focused using Ollama.

Efficient Vector Search: Uses Facebook AI Similarity Search (FAISS) for lightning-fast document retrieval.

Smart Chunking: Automatically breaks down large PDFs into manageable chunks with estimated page tracking.

Persistence: Saves processed vectors and text chunks locally so you don't have to re-process the same PDF twice.

Architecture diagram

PDF → Chunking → Embeddings (nomic-embed-text) → FAISS Index
Query → Embed → FAISS Search → Top-3 Chunks → Mistral → Answer

🛠️ Tech Stack
LLM & Embeddings: Ollama (Models: mistral and nomic-embed-text)

Vector Database: FAISS

PDF Processing: PyPDF2

Language: Python 3.x

📋 Prerequisites
Install Ollama: Download and install from ollama.com.

Pull Required Models:

ollama pull mistral
ollama pull nomic-embed-text
⚙️ Installation

Install dependencies:


pip install ollama faiss-cpu PyPDF2 numpy

📂 Project Structure
main.py: The central entry point to run the application.

pdf_to_vector.py: Handles PDF text extraction, chunking, embedding generation, and FAISS index creation.

question_vector.py: Manages the similarity search and the generation of answers using the retrieved context.

vectors.index: (Generated) The FAISS vector database.

chunks.pkl: (Generated) Pickled text data and metadata.

🚀 Usage
Run the main interface:


python main.py

Options:
Process new PDF: Select option 1, then enter the path to your PDF file (e.g., data/my_doc.pdf). This will create the vectors.index and chunks.pkl files.

Ask a question: Select option 2 to chat with the previously processed document. The system will retrieve the 3 most relevant chunks to formulate an answer.

🧠 How it Works
Ingestion: The system reads a PDF and splits the text into chunks of ~500 characters.

Embedding: Each chunk is converted into a high-dimensional vector using the nomic-embed-text model.

Indexing: These vectors are stored in a FAISS index using Inner Product (IP) similarity.

Retrieval: When you ask a question, your query is embedded, and FAISS finds the top-3 closest matching chunks.

Generation: The retrieved chunks are fed into the mistral model as "Context," which then provides an answer based strictly on that information.