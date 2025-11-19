def build_prompt(docs, metas, question):
    context = ""
    for d, m in zip(docs, metas):
        context += f"[{m['source']} - page {m['page']}]\n{d}\n\n"

    system = (
        "You must answer ONLY using the verified context from DOH/WHO. "
        "If the answer is not in the context, say you do not know. No hallucinations."
    )

    return f"{system}\n\nCONTEXT:\n{context}\nQUESTION:\n{question}\nANSWER:"
