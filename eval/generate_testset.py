import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
import ollama
import faiss
import numpy as np

# Step 1: Load your saved chunks

with open("chunks.pkl","rb") as f:
    data = pickle.load(f)

chunks = data['chunks']
metadata = data['metadata']

# Step 2: Preview chunks so you can write good questions

print("=== SAMPLE CHUNKS — read these, then write questions based on them ===\n")
for i, chunk in enumerate(chunks[:10]):
    print(f"--- Chunk {i} ---")
    print(chunk[:300])
    print()

# Step 3: Load the FAISS index
index = faiss.read_index("vectors.index")

# Step 4: A function that runs your RAG and returns BOTH the answer AND the raw chunks

def get_answer_and_contexts(question):
    #Embed the question
    response = ollama.embed(model="nomic-embed-text", input=question)
    query_vector = np.array(response['embeddings'][0]).astype("float32").reshape(1, -1) 

    print(query_vector)
    # Search FAISS
    scores, indices = index.search(query_vector, 3)

    # Collect raw chunk texts — THIS is what RAGAS needs as 'contexts'
    retrieved_contexts = []
    for idx in indices[0]:
        retrieved_contexts.append(chunks[idx])

    # Build the prompt exactly like your existing code does
    context_parts = []
    for score, idx in zip(scores[0], indices[0]):
        page_num = metadata[idx]['estimated_page']
        context_parts.append(f"[Page {page_num}: {chunks[idx]}]")

    context = "\n\n".join(context_parts)

    full_prompt = f"""
    Use the following context to answer the user's question.
    If the answer is not in the context, say you don't know.
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:
    """

    gen_response = ollama.generate(model="mistral", prompt=full_prompt)
    answer = gen_response["response"]

    # Return BOTH — the answer for RAGAS, and the contexts for RAGAS
    return answer, retrieved_contexts    


# Step 5: Write your test questions HERE after reading the chunk preview above
test_questions = [
    "What are the three steps of the RAG architecture?",
    "What is the recommended chunk size according to the document?",
    "What makes AI Agents different from standard chatbots?",
    "What are the three metrics in the RAG Triad?",
    "What embedding models does the document mention?"                        # write 5-10 based on YOUR actual PDF
]

# Step 6: Run the pipeline for each question and collect everything
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}


for question in test_questions:
    print(f"Running: {question}")
    
    answer, contexts = get_answer_and_contexts(question)
    
    eval_data["question"].append(question)
    eval_data["answer"].append(answer)
    eval_data["contexts"].append(contexts)
    eval_data["ground_truth"].append("FILL_THIS_IN")

    print(f"  Done. Answer starts with: {answer[:80]}...")

# Step 7: Save to a JSON file so you can fill in ground truths manually
os.makedirs("eval/results", exist_ok=True)


with open("eval/results/testset_raw.json", "w") as f:
    json.dump(eval_data, f, indent=2)

print("Done. Now open eval/results/testset_raw.json")
print("Fill in every 'FILL_THIS_IN' with the correct answer, then run run_eval.py")