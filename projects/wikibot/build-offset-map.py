# Copyright 2025 Brian Mosely. This is NOT open source software and is presented for educational purposes.

# Run this ONCE after merging wiki_metadata.jsonl
def build_metadata_offset_index(metadata_path, index_path):
    with open(metadata_path, "r") as meta_file, open(index_path, "w") as index_file:
        offset = 0
        for line in meta_file:
            index_file.write(f"{offset}\n")
            offset += len(line)

# Usage
build_metadata_offset_index("./embeddings/wiki_metadata.jsonl", "./embeddings/wiki_metadata.idx")