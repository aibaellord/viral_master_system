"""
SemanticMemory: Local vector database for content memory, semantic search, and retrieval-augmented generation.
- Integrates ChromaDB or FAISS for high-speed, local semantic search.
- Stores and retrieves embeddings for smarter, context-aware content.
"""
import logging
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

class SemanticMemory:
    def __init__(self, collection_name: str = "content_memory"):
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        if chromadb:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(collection_name)
            self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.client = None
            self.collection = None
            self.embed_fn = None

    def add_content(self, content_id: str, text: str, metadata: Dict[str, Any] = None):
        if self.collection and self.embed_fn:
            emb = self.embed_fn(text)
            self.collection.add(documents=[text], embeddings=[emb], metadatas=[metadata or {}], ids=[content_id])
            self.logger.info(f"Added content {content_id} to semantic memory.")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.collection and self.embed_fn:
            emb = self.embed_fn(query_text)
            results = self.collection.query(query_embeddings=[emb], n_results=top_k)
            return results.get("documents", [])
        return []
