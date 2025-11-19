from utils import build_prompt

class RAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query, top_k=4):
        results = self.retriever.query(query, top_k)

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # build prompt
        prompt = build_prompt(docs, metas, query)

        # llm generate
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": metas
        }
