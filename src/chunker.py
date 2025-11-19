class TextChunker:
    def __init__(self, size=2000, overlap=200):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str):
        if len(text) <= self.size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - self.overlap
        return chunks
