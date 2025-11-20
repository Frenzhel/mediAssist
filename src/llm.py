import os
import requests

class LLM:
    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3")

    def generate(self, prompt):
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0
        }

        r = requests.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

        # Supports older and newer Ollama formats
        if "response" in data:
            return data["response"]

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return str(data)
