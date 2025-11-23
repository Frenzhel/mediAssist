import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.retriever import VectorStore
from src.llm import LLM
from src.rag import RAGPipeline
from src.embedder import Embedder

load_dotenv()

class Query(BaseModel):
    question: str

app = FastAPI(title="Health RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_DIR = os.getenv("CHROMA_DIR", "./vector_db")
retriever = VectorStore(VECTOR_DIR)
embedder = Embedder()
llm = LLM()
pipeline = RAGPipeline(
    retriever=retriever, 
    llm=llm, 
    embedder=embedder, 
    top_k=int(os.getenv("TOP_K", "4"))
)

@app.post("/chat")
def chat(req: Query):
    return pipeline.answer(req.question)
