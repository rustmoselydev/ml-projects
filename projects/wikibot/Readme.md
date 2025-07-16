Copyright 2025 Brian Mosely. This is NOT open source software and is presented for educational purposes.

To make use of this model for offline RAG on any text content:

- Chunk the data with parquet-chunks.py. If your data is in a different format, you'll need to adjust. This script originally works from letter-separated parquet files that are a Wikipedia dump and puts them into letter-separated .jsonl files
- Run faiss-index.py to generate embeddings and metadata for retrieval later
- Run build-offset-map.py to build the offset map for the retrieval model to reference
- Run combine-metadata.sh to take the letter .jsonl files and combine them into one
- Choose either wikibot-llama or wikibot-tinyllama depending on your preferences. llama.cpp doesn't have mps support, so run tinyllama if you're on mac.
