"""
rag.py
Professor-style RAGPipeline that matches the classroom architecture,
but calls the retriever with a query embedding (modern flow).
"""

from src.utils import build_prompt

class RAGPipeline:
    def __init__(self, retriever, llm, embedder, top_k=4):
        """
        retriever: VectorStore instance
        llm: LLM instance
        embedder: Embedder instance
        """
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.top_k = top_k

    def answer(self, question: str, top_k: int = None):
        if top_k is None:
            top_k = self.top_k

        # 1) embed the user query
        q_emb = self.embedder.embed([question])[0]  # single vector

        # 2) retrieve top-k chunks by embedding similarity
        results = self.retriever.query_by_embedding(q_emb, n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        # 3) build prompt following professor-style prompt engineering
        prompt = build_prompt(docs, metas, question)

        # 4) call LLM
        answer = self.llm.generate(prompt)

        return {"answer": answer, "sources": metas}
