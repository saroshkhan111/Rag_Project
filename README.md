# RAG Daily Workflow Agent

AI agent automating daily tasks using 
RAG pipeline.

## Features
- Calendar reminders automation
- Watchlist management
- Daily item monitoring
- Confidence thresholding (hallucination prevention)
- TTL-based re-indexing for data freshness

## Tech Stack
- Python
- FAISS (Vector Store)
- FastAPI
- LLM Integration
- Embeddings (sentence-transformers)

## Architecture
User Query → Embedding → FAISS Search 
→ Top 5 Chunks → LLM → Response
