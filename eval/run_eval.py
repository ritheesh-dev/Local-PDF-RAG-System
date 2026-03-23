import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.run_config import RunConfig
# Step 1: Load your filled testset
with open("eval/results/testset_raw.json", "r") as f:
    eval_data = json.load(f)

# Step 2: Check that you actually filled in the ground truths
unfilled = [i for i, g in enumerate(eval_data["ground_truth"]) if g == "FILL_THIS_IN"]

if unfilled:
    print(f"You haven't filled ground truth for questions at positions: {unfilled}")
    print("Open eval/results/testset_raw.json and fill them in first.")
    sys.exit(1)

# Step 3: Connect RAGAS to your local Ollama
print("Connecting to Ollama...")

ollama_llm = LangchainLLMWrapper(
    Ollama(model="mistral")
)

ollama_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)

# Step 4: Convert your dictionary to a HuggingFace Dataset
dataset = Dataset.from_dict(eval_data)

# Step 5: Run the evaluation
print("Running evaluation — this will take a few minutes...")
print("RAGAS is using Mistral internally to judge each answer.\n")

# Tell RAGAS to be patient and go one at a time
run_config = RunConfig(
    timeout=180,        # wait up to 3 minutes per LLM call
    max_workers=1,      # only one job at a time — Ollama can't handle parallel
    max_retries=2,      # if a call fails, try again twice before giving up
)
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ],
    llm=ollama_llm,
    embeddings=ollama_embeddings,
    run_config=run_config,
)

# Step 6: Print the scores clearly
import numpy as np

print("\n=== YOUR RAGAS SCORES ===\n")

for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']:
    score = results[metric]
    
    # Handle both old RAGAS (float) and new RAGAS (list)
    if isinstance(score, list):
        avg = sum(score) / len(score)
    else:
        avg = score
    
    print(f"{metric:<22}: {avg:.3f}")

# Step 7: Print what the scores mean so you don't have to memorise
print("\n=== WHAT YOUR SCORES MEAN ===\n")
print("Faithfulness      — is Mistral only using your chunks? (hallucination check)")
print("Answer relevancy  — is the answer actually addressing the question?")
print("Context recall    — did FAISS retrieve chunks that contain the answer?")
print("Context precision — were the retrieved chunks relevant, not noisy?")

print("\n=== SCORE GUIDE ===")
print("0.8 and above → solid")
print("0.6 to 0.8    → acceptable, but look at which questions scored low")
print("below 0.6     → something is broken, check that metric specifically")

# Step 8: Save the detailed breakdown to CSV
df = results.to_pandas()
os.makedirs("eval/results", exist_ok=True)
df.to_csv("eval/results/ragas_scores.csv", index=False)

print("\nDetailed scores saved to eval/results/ragas_scores.csv")
print("Open it to see scores per question — not just the average.")