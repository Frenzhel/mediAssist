from chromadb import PersistentClient

class VectorStore:
    def __init__(self, persist_path: str):
        self.client = PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(name="health_docs")

    def add(self, ids, documents, metadatas, embeddings):
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def query_by_embedding(self, query_embedding, n_results=4, include=None):
        if include is None:
            include = ["documents", "metadatas", "distances"]
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include
        )

    def close(self):
        try:
            self.client.persist()
            self.client.shutdown()
        except Exception:
            pass
