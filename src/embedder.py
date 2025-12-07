from sentence_transformers import SentenceTransformer
import os

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        local_path = os.path.join("model_cache", model_name)

        if not os.path.exists(local_path):
            raise RuntimeError(
                f"❌ OFFLINE MODE: Model not found at {local_path}. "
                f"Download it once with internet and place it in model_cache/{model_name}/"
            )

        self.model = SentenceTransformer(
            local_path,
            cache_folder="model_cache",
            trust_remote_code=True
        )

    def embed(self, texts):
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
