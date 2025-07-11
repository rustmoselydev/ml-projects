# from fastapi import FastAPI, Request
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_metadata_index(index_path):
    with open(index_path, "r") as f:
        return [int(line.strip()) for line in f]

def fetch_metadata_by_index(index_list, metadata_path, offset_map):
    results = []
    with open(metadata_path, "r") as f:
        for i in index_list:
            f.seek(offset_map[i])
            line = f.readline()
            results.append(json.loads(line))
    return results

# Load FAISS index
index = faiss.read_index("./embeddings/wiki_ivf.index", faiss.IO_FLAG_MMAP)
index.nprobe = 20
print("Loaded index")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")

# Load offset map
offset_map = load_metadata_index("./embeddings/wiki_metadata.idx")
print("Loaded metadata index")

# Model config
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load LLM and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32
)

llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

def retrieve_relevant_chunks(query, k=10):
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    return fetch_metadata_by_index(indices[0], "./embeddings/wiki_metadata.jsonl", offset_map)

def generate_answer(query):
    relevant_chunks = retrieve_relevant_chunks(query)
    context = "\n".join([f"[{chunk['title']}]\n{chunk['text']}" for chunk in relevant_chunks])
    context = context[:2048]  # set to LLM's max input limit
    return f"Here are the most relevant entries:\n\n{context}"

# Prompt Template
def build_prompt(query, context):
    return f"""<s>[INST] <<SYS>>
You are a helpful assistant. Use the following context to answer the user's question.
<</SYS>>

Context:
{context}

Question: {query} [/INST]
"""

# CLI loop
print("""
      =================================
      #------------WIKIBOT------------#
      =================================

      (Ctrl+C to quit)
      """)
while True:
    try:
        user_query = input("You: ")
        context = generate_answer(user_query)
        prompt = build_prompt(user_query, context)

        response = llm(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)[0]["generated_text"]
        answer = response[len(prompt):].strip()  # Remove the prompt from the output

        print(f"\nChatbot: {answer}\n")

    except KeyboardInterrupt:
        print("\nExiting.")
        break