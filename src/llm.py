import requests
import os

class LLM:
    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST")
        self.model = os.getenv("OLLAMA_MODEL")

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0
        }
        r = requests.post(f"{self.host}/api/generate", json=payload)
        r.raise_for_status()
        return r.json()["response"]
