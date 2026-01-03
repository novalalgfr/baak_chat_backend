import os
import torch
import uvicorn
import chromadb
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from groq import Groq

# Import dari folder-folder yang sudah dipecah
from app.core.config import MODEL_NAME, DB_PATH, COLLECTION_NAME, GROQ_API_KEY
from app.core.state import resources
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chroma_service import retrieve_knowledge
from app.services.llm_service import generate_answer

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ [Init] Loading Resources...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Loading model on: {device.upper()}")
    
    # 1. Load Embedding Model
    resources['embedding_model'] = SentenceTransformer(MODEL_NAME, device=device)
    
    # 2. Load ChromaDB
    if os.path.exists(DB_PATH):
        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            resources['collection'] = client.get_collection(COLLECTION_NAME)
            print("✅ Connected to ChromaDB.")
        except:
            print("⚠️ Collection not found. Run 'build_db.py' first.")
    else:
        print("⚠️ Database path missing. Run 'build_db.py' first.")
    
    # 3. Load Groq
    resources['groq_client'] = Groq(api_key=GROQ_API_KEY)
    print("✅ BAAK Assistant Ready!")

    yield

    resources.clear()
    print("🛑 Shutting down Resources...")

app = FastAPI(title="BAAK AI Service", version="4.0-Enterprise", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def chat_with_baak(request: ChatRequest):
    user_query = request.question
    print(f"\nUser Query: {user_query}")
    
    # 1. Cari Data (Panggil Service Chroma)
    context_docs = retrieve_knowledge(user_query, top_k=12)
    sources = list(set([doc['source'] for doc in context_docs]))
    
    if context_docs:
        print(f"Top Source: {context_docs[0]['source']}")
    else:
        return ChatResponse(
            answer="Mohon maaf, saya tidak menemukan informasi terkait di database saat ini.",
            sources=[]
        )

    # 2. Generate Jawaban (Panggil Service LLM)
    bot_answer = generate_answer(user_query, context_docs)
    
    return ChatResponse(
        answer=bot_answer,
        sources=sources
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)