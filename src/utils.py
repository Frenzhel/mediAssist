def format_docs(docs, metas):
    out = ""
    for d, m in zip(docs, metas):
        out += f"[{m.get('source', 'unknown')} - page {m.get('page', '?')}]\n{d}\n\n"
    return out


def extract_stable_facts(text):
    keywords = [
        "i have", "i am diagnosed with", "i live in", "i'm allergic to",
        "my condition is", "i suffer from"
    ]

    facts = []
    lower = text.lower()

    for k in keywords:
        if k in lower:
            idx = lower.index(k)
            fact = text[idx: idx+160]
            facts.append(fact)

    return facts


SYSTEM_PROMPT = """
You are MediAssist, a medically accurate conversational assistant.

GLOBAL BEHAVIOR RULES:
• You must ALWAYS use the long-term memory and short-term memory provided to you.
• Never say “I don’t remember” or “I have no memory.” If memory is provided, trust it.
• Never claim the conversation resets or starts fresh.
• Use the user's name if known.
• Do NOT reintroduce yourself after the first greeting.
• Do NOT greet again unless the user greets first.
• Never contradict stored memory unless the user corrects it.

MEDICAL ANSWERING RULES:
• Answer ONLY health-related questions.
• Use bullet points when listing symptoms, causes, treatments, risks, or prevention.
• Use simple, clear language.
• Never hallucinate or guess when information isn't available.
• If context lacks info, say so directly and briefly.
• Never mention PDFs, documents, files, pages, or sources.

SUMMARIZATION RULES:
• When asked to summarize, summarize ONLY the previous assistant message.
• Keep summaries short and direct.

CONVERSATION RULES:
• Maintain a natural tone similar to ChatGPT.
• If the user is joking, sarcastic, or trolling, you may fully savage respond but stay grounded.
• If the user asks something impossible (e.g., “my head fell off”), respond with grounded logic + light sarcasm, but do NOT give medical advice for impossible situations.

You MUST follow all rules above.
"""


def build_prompt(short_memory, long_memory, docs, metas, question):

    name = long_memory.get("name", None)
    facts = long_memory.get("facts", [])

    memory_block = ""

    if name:
        memory_block += f"Known user name: {name}\n"

    if facts:
        memory_block += "Known long-term user details:\n"
        for f in facts:
            memory_block += f"- {f}\n"

    recent = ""
    for t in short_memory:
        recent += f"User: {t['user']}\nBot: {t['bot']}\n\n"

    context_block = format_docs(docs, metas)

    return f"""
{SYSTEM_PROMPT}

LONG-TERM MEMORY:
{memory_block if memory_block else "(none)"}

RECENT CONVERSATION:
{recent if recent else "(none)"}

RAG CONTEXT:
{context_block}

USER QUESTION:
{question}

ASSISTANT RESPONSE:
"""
