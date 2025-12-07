from src.utils import build_prompt, extract_stable_facts
import re

class RAGPipeline:
    def __init__(self, retriever, llm, embedder, top_k=4):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.top_k = top_k

        self.short_memory = []
        self.long_memory = {
            "name": None,
            "facts": []
        }

        self.max_short = 6
        self.greeted = False

    def detect_name(self, text):
        text_low = text.lower().strip()

        intro_keywords = [
            "my name is",
            "i am",
            "i'm"
        ]

        invalid_patterns = [
            "in half", "bleeding", "hurt", "injured", "cut", "sliced",
            "broken", "dead", "dying", "exploded", "torn", "ripped",
            "my head", "my arm", "my leg"
        ]

        for bad in invalid_patterns:
            if bad in text_low:
                return None

        for key in intro_keywords:
            if key in text_low:
                after = text_low.split(key, 1)[1].strip()
                candidate = after.split(" ")[0]

                if not candidate.isalpha():
                    return None
                if len(candidate) < 2:
                    return None
                return candidate.capitalize()

        return None

    def update_memory(self, user_msg, bot_msg):
        self.short_memory.append({"user": user_msg, "bot": bot_msg})
        if len(self.short_memory) > self.max_short:
            self.short_memory.pop(0)

        name = self.detect_name(user_msg)
        if name:
            self.long_memory["name"] = name

        facts = extract_stable_facts(bot_msg)
        for f in facts:
            if f not in self.long_memory["facts"]:
                self.long_memory["facts"].append(f)

    def summarize_last_bot_message(self):
        last = self.short_memory[-1]["bot"]
        prompt = (
            "Summarize the following text in 1–3 simple sentences:\n\n"
            f"{last}"
        )
        return self.llm.generate(prompt)

    def detect_nonsense(self, text):
        text_low = text.lower()

        impossible_keywords = [
            "head was torn off",
            "my head came off",
            "my brain fell out",
            "my heart stopped",
            "i died",
            "i am dead",
            "my arm exploded",
            "i lost all my blood"
        ]

        for k in impossible_keywords:
            if k in text_low:
                return (
                    "If your head was torn off, the only thing you'd be doing is being dead. "
                    "Try again, this time with a functioning brain."
                )
        return None


    def handle_special_cases(self, question):
        q = question.lower().strip()
        if q in ["summarize that", "can you summarize that?", "can you summarize that"]:
            if len(self.short_memory) == 0:
                return "There's nothing to summarize yet."
            return self.summarize_last_bot_message()

        if "remember my name" in q:
            if self.long_memory["name"]:
                return f"Yes, your name is {self.long_memory['name']}."
            else:
                return "I don’t know your name yet, but you can tell me anytime."

        return None

    def answer(self, question: str, top_k: int = None):
        name = self.detect_name(question)
        if name:
            self.long_memory["name"] = name

        special = self.handle_special_cases(question)
        if special:
            bot = special
            self.update_memory(question, bot)
            return {"answer": bot, "sources": []}

        is_greeting = question.lower() in ["hi", "hello", "hey", "good morning", "good afternoon"]
        name = self.long_memory["name"]

        if is_greeting and not self.greeted:
            self.greeted = True
            if name:
                bot = f"Hi {name}, how can I help you today?"
            else:
                bot = "Hello! How can I assist you today?"
            self.update_memory(question, bot)
            return {"answer": bot, "sources": []}

        if is_greeting and self.greeted:
            if name:
                bot = f"Hi {name}, what would you like to talk about?"
            else:
                bot = "Hi again. How can I help?"
            self.update_memory(question, bot)
            return {"answer": bot, "sources": []}

        nonsense = self.detect_nonsense(question)
        if nonsense:
            bot = nonsense
            self.update_memory(question, bot)
            return {"answer": bot, "sources": []}

        if top_k is None:
            top_k = self.top_k

        q_emb = self.embedder.embed([question])[0]
        results = self.retriever.query_by_embedding(q_emb, n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        final_prompt = build_prompt(
            short_memory=self.short_memory,
            long_memory=self.long_memory,
            docs=docs,
            metas=metas,
            question=question
        )

        bot = self.llm.generate(final_prompt)

        self.update_memory(question, bot)

        return {"answer": bot, "sources": metas}
