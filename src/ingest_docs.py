import os
from dotenv import load_dotenv
load_dotenv()

from src.loader import PDFLoader
from src.chunker import TextChunker
from src.embedder import Embedder
from src.retriever import VectorStore

DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTOR_DIR = os.getenv("CHROMA_DIR", "./vector_db")

def ingest_all():
    loader = PDFLoader()
    chunker = TextChunker()
    embedder = Embedder()
    store = VectorStore(VECTOR_DIR)

    ids = []
    docs = []
    metas = []
    all_texts_for_embedding = []

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(DATA_DIR, filename)
        print(f"Loading PDF -> {filename}")
        pages = loader.load(path)
        for p in pages:
            chunks = chunker.chunk(p["text"])
            for idx, chunk in enumerate(chunks):
                uid = f"{filename}::p{p['page']}::c{idx}"
                ids.append(uid)
                docs.append(chunk)
                metas.append({"source": filename, "page": p["page"], "chunk_index": idx})
                all_texts_for_embedding.append(chunk)

    if not docs:
        print("No documents found to ingest.")
        return

    print(f"Total chunks to embed: {len(docs)}")
    embeddings = embedder.embed(all_texts_for_embedding)

    print("Saving to vector database...")
    store.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    store.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_all()
