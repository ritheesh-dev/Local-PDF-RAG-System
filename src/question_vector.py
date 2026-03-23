import ollama
import faiss
import numpy as np
import pickle
import os


def ask_question(question):
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("Vector DB is not found")
        print("Please run 'pdf_to_vector.py'")
        return None

    try:
        index = faiss.read_index("vectors.index")

        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']

        # Create embedding
        response = ollama.embed(
            model="nomic-embed-text",
            input=question
        )

        query_vector = np.array(
            response['embeddings'][0]
        ).astype("float32").reshape(1, -1)

        scores, indices = index.search(query_vector, 3)

        print(f"Found {len(indices[0])} relevant chunks:")

        context_parts = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            page_num = metadata[idx]['estimated_page']
            print(f"Chunk {i+1}: Score {score:.3f} (~ Page {page_num})")

            chunk_text = chunks[idx]
            context_parts.append(f"[Page {page_num}: {chunk_text}]")

        context = "\n\n".join(context_parts)

        full_prompt = f"""
        Use the following context to answer the user's question.
        If the answer is not in the context, say you don't know.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """

        # LLM for answer
        response = ollama.generate(
            model="mistral",
            prompt=full_prompt
        )

        return response["response"]

    except Exception as e:
        print(f"Error in question processing: {e}")
        return None



