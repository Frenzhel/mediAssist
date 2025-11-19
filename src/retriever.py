import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_dir):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir
        ))
        self.collection = self.client.get_or_create_collection("health_docs")

    def add(self, ids, documents, metadatas):
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, q, top_k):
        return self.collection.query(
            query_texts=[q],
            n=top_k,
            include=["documents", "metadatas"]
        )
