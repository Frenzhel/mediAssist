import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from retriever import VectorStore
from llm import LLM
from rag import RAGPipeline

load_dotenv()

class Query(BaseModel):
    question: str

app = FastAPI()

# init components
retriever = VectorStore(os.getenv("CHROMA_DIR"))
llm = LLM()
pipeline = RAGPipeline(retriever, llm)

@app.post("/chat")
def chat(req: Query):
    return pipeline.answer(req.question)
