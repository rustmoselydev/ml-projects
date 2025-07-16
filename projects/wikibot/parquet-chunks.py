# Copyright 2025 Brian Mosely. This is NOT open source software and is presented for educational purposes.
# Turn the data into smaller chunks for embedding

import os
import json
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

# Sentence-aware chunking with a junk filter
def sentence_chunk(text, max_words=256, overlap_words=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if current_len + len(sentence_words) > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # Start next chunk with overlap
                overlap = current_chunk[-overlap_words:] if len(current_chunk) > overlap_words else current_chunk
                current_chunk = list(overlap)
                current_len = len(current_chunk)
        current_chunk.extend(sentence_words)
        current_len += len(sentence_words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [chunk for chunk in chunks if len(chunk.split()) > 20]  # filter out trivial chunks

# Process all parquet files
os.makedirs("./chunks", exist_ok=True)

for file in os.listdir("./archive"):
    if not file.endswith(".parquet"):
        continue

    print(f"Processing {file}")
    df = pd.read_parquet(f"./archive/{file}", columns=["id", "title", "text"])
    output = []

    for _, row in df.iterrows():
        chunks = sentence_chunk(row["text"])
        for chunk in chunks:
            output.append({
                "id": row["id"],
                "title": row["title"],
                "text": chunk
            })

    # Save json chunked output
    with open(f"./chunks/{file}.json", "w") as f:
        json.dump(output, f, indent=2)