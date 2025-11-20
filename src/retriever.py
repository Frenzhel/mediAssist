"""
retriever.py
Professor-style VectorStore wrapper but using modern Chroma PersistentClient.
"""

from chromadb import PersistentClient

class VectorStore:
    def __init__(self, persist_path: str):
        """
        Initialize a persistent local Chroma client.
        persist_path: path to store the DB files (e.g., "./vector_db")
        """
        # PersistentClient accepts a path to a folder
        self.client = PersistentClient(path=persist_path)
        # keep collection name simple and stable for grading
        self.collection = self.client.get_or_create_collection(name="health_docs")

    def add(self, ids, documents, metadatas, embeddings):
        """
        Add documents with precomputed embeddings to the collection.
        ids: list of string ids
        documents: list of text chunks
        metadatas: list of metadata dicts
        embeddings: list of vector embeddings (same order)
        """
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def query_by_embedding(self, query_embedding, n_results=4, include=None):
        """
        Query using a precomputed embedding.
        Returns whatever collection.query returns (dict).
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include
        )

    def close(self):
        # Close persistent client if needed
        try:
            self.client.persist()
            self.client.shutdown()
        except Exception:
            pass
