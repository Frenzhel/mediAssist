from src.utils import build_prompt

class RAGPipeline:
    def __init__(self, retriever, llm, embedder, top_k=4):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.top_k = top_k

    def answer(self, question: str, top_k: int = None):
        if top_k is None:
            top_k = self.top_k

        q_emb = self.embedder.embed([question])[0]

        results = self.retriever.query_by_embedding(q_emb, n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        prompt = build_prompt(docs, metas, question)

        answer = self.llm.generate(prompt)

        return {"answer": answer, "sources": metas}
