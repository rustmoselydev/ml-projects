# Copyright 2025 Brian Mosely. This is NOT open source software and is presented for educational purposes.

from llama_cpp import Llama
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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

# Quantized LLaMA 3 8B model
MODEL_PATH = "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

# Initialize LLaMA model with context size and Metal acceleration
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=6400,
    n_threads=6,        # Adjust based on CPU
    n_gpu_layers=0,    # Adjust if on CUDA- this model isn't MPS friendly
    verbose=False
)


def retrieve_relevant_chunks(query, k=20):
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    return fetch_metadata_by_index(indices[0], "./embeddings/wiki_metadata.jsonl", offset_map)

def generate_answer(query):
    relevant_chunks = retrieve_relevant_chunks(query)
    context = "\n".join([f"[{chunk['title']}]\n{chunk['text']}" for chunk in relevant_chunks])
    context = context[:6400]  # set to LLM's max input limit
    return f"Here are the most relevant entries:\n\n{context}"

# FastAPI app
# app = FastAPI()

# @app.post("/chat")
def chat(query):
    if not query:
        return {"error": "Missing query"}

    # Retrieve relevant Wikipedia context
    context = generate_answer(query)

    # Construct prompt for LLaMA
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant who answers questions using provided information from Wikipedia.
<|start_header_id|>user<|end_header_id|>
{query}
<|start_header_id|>assistant<|end_header_id|>
Relevant Wikipedia passages:
{context}"""

    # Generate answer
    response = llm(prompt, max_tokens=1024, temperature=0.7)
    answer = response["choices"][0]["text"].strip()

    return answer

# CLI loop
print("Ask me anything (Ctrl+C to quit):")
while True:
    try:
        user_query = input("You: ").strip()
        answer = chat(user_query)
        print(f"\nChatbot:\n{answer}\n")
    except KeyboardInterrupt:
        print("\nExiting.")
        break