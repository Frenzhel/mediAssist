def build_prompt(docs, metas, question):
    context = ""
    for d, m in zip(docs, metas):
        context += f"[{m.get('source','unknown')} - page {m.get('page','?')}]\n{d}\n\n"

    system = (
        "You are a precise assistant. Answer only using the provided verified context "
        "from DOH/WHO documents. If the answer is not present, say you don't know. No hallucinations."
    )
    return f"{system}\n\nCONTEXT:\n{context}\nQUESTION:\n{question}\nANSWER:"
