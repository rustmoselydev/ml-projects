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
model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")

# Load offset map
offset_map = load_metadata_index("./embeddings/wiki_metadata.idx")
print("Loaded metadata index")


def retrieve_relevant_chunks(query, k=20):
    query_embedding = model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    return fetch_metadata_by_index(indices[0], "./embeddings/wiki_metadata.jsonl", offset_map)

def generate_answer(query):
    relevant_chunks = retrieve_relevant_chunks(query)
    context = "\n".join([f"[{chunk['title']}]\n{chunk['text']}" for chunk in relevant_chunks])
    # context = context[:4096]  # set to LLM's max input limit
    return f"Here are the most relevant entries:\n\n{context}"

# CLI loop
print("Ask me anything (Ctrl+C to quit):")
while True:
    try:
        user_query = input("You: ").strip()
        answer = generate_answer(user_query)
        print(f"\nChatbot:\n{answer}\n")
    except KeyboardInterrupt:
        print("\nExiting.")
        break