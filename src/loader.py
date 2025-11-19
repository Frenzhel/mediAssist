import pdfplumber
import os
import re

class PDFLoader:
    def load(self, path: str):
        pages = []
        filename = os.path.basename(path)

        with pdfplumber.open(path) as pdf:
            for num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = re.sub(r"\n{2,}", "\n", text).strip()
                if text:
                    pages.append({
                        "page": num,
                        "text": text,
                        "source": filename
                    })
        return pages
