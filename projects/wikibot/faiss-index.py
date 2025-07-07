# Create a faiss index and metadata to use as embeddings for the chatbot
import json, os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Config ---
chunk_dir = "./chunks"
index_path = "./embeddings/wiki_ivf.index"
metadata_dir = "./embeddings/metadata_chunks"
processed_log_path = "./embeddings/processed_files.txt"
batch_size = 500
nlist = 256  # Number of clusters for IVF

# --- Setup ---
os.makedirs(metadata_dir, exist_ok=True)
model = SentenceTransformer('all-MiniLM-L6-v2', device="mps")
dimension = model.get_sentence_embedding_dimension()

# --- Initialize Index ---
if os.path.exists(index_path):
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    print("Loaded existing FAISS IVF index.")
else:
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.nprobe = 10
    print("Created new FAISS IVF index (not yet trained).")

# --- Load Processed Files ---
if os.path.exists(processed_log_path):
    with open(processed_log_path, "r") as f:
        processed_files = set(f.read().splitlines())
else:
    processed_files = set()

# --- Training if needed ---
if not index.is_trained:
    print("Collecting training data for IVF...")
    training_samples = []
    files = sorted(os.listdir(chunk_dir))
    for file_name in files:
        if len(training_samples) > 5000:
            break
        with open(os.path.join(chunk_dir, file_name), "r") as f:
            data = json.load(f)
            for item in data[:100]:  # limit per file
                training_samples.append(model.encode(item["text"]))
    training_array = np.array(training_samples).astype(np.float32)
    index.train(training_array)
    print("Index trained.")

# --- Indexing Loop ---
chunk_files = sorted(os.listdir(chunk_dir))
for file_name in chunk_files:
    if file_name in processed_files:
        print(f"Skipping {file_name}")
        continue

    file_path = os.path.join(chunk_dir, file_name)
    print(f"Processing {file_name}")
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

    metadata_out = os.path.join(metadata_dir, f"{file_name}.jsonl")
    with open(metadata_out, "w") as meta_file:
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            texts = [item["text"] for item in batch]
            embeddings = model.encode(texts, show_progress_bar=False)

            embeddings_np = np.array(embeddings).astype(np.float32)
            index.add(embeddings_np)

            for item in batch:
                meta_file.write(json.dumps({
                    "id": item["id"],
                    "title": item["title"],
                    "text": item["text"]
                }) + "\n")

            print(f"Added {i+len(batch)}/{len(data)} from {file_name}")
    index.make_direct_map()
    faiss.write_index(index, index_path)
    
    with open(processed_log_path, "a") as log:
        log.write(file_name + "\n")
    print(f"Completed {file_name}\n")

print("All chunk files processed.")