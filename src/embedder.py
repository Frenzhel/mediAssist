"""
embedder.py
Simple wrapper around sentence-transformers to produce vector embeddings.
"""

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """
        texts: list[str]
        returns: list[list[float]] (or numpy array) - compatible with Chroma
        """
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.tolist()
