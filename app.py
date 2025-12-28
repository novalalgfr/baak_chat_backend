import os
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'chroma_db')
COLLECTION_NAME = "baak_knowledge"
MODEL_NAME = "intfloat/multilingual-e5-small"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="BAAK AI Service", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = None
vector_db_collection = None
groq_client = None

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.on_event("startup")
def load_resources():
    global embedding_model, vector_db_collection, groq_client
    
    print("⏳ [Init] Memuat Embedding Model (multilingual-e5-small)...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    print("⏳ [Init] Menghubungkan ke Vector DB (ChromaDB)...")
    if os.path.exists(DB_PATH):
        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            vector_db_collection = client.get_collection(COLLECTION_NAME)
            print(f"   -> Terhubung ke koleksi: {COLLECTION_NAME}")
        except:
            print("   -> ⚠️ Koleksi tidak ditemukan. Harap jalankan 'build_db.py' dahulu.")
    else:
        print("   -> ⚠️ Database belum dibuat. Harap jalankan 'build_db.py' dahulu.")
    
    print("⏳ [Init] Menghubungkan ke Groq Cloud...")
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    print("✅ [Ready] Sistem BAAK AI Siap Melayani!")

def retrieve_knowledge(query: str, top_k: int = 20):
    if not vector_db_collection:
        return []

    query_vector = embedding_model.encode([f"query: {query}"]).tolist()
    
    results = vector_db_collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )
    
    documents = []
    if results['documents']:
        for i in range(len(results['documents'][0])):
            clean_content = results['documents'][0][i].replace("passage: ", "")
            meta = results['metadatas'][0][i]
            
            documents.append({
                "content": clean_content,
                "source": meta.get('source', 'unknown'),
                "type": meta.get('type', 'unknown')
            })
    return documents

@app.post("/chat", response_model=ChatResponse)
def chat_with_baak(request: ChatRequest):
    user_query = request.question
    
    print(f"\n🔍 [User Query]: {user_query}")
    context_docs = retrieve_knowledge(user_query, top_k=20)
    
    sources = [doc['source'] for doc in context_docs]
    
    if not context_docs:
        return ChatResponse(
            answer="Mohon maaf, sistem database BAAK belum siap atau tidak menemukan data terkait.",
            sources=[]
        )

    context_text = ""
    for doc in context_docs:
        context_text += f"- [{doc['type'].upper()}] {doc['content']} (File: {doc['source']})\n"

    system_prompt = """
    Kamu adalah Asisten Virtual Cerdas untuk BAAK (Biro Administrasi Akademik) Universitas Gunadarma.
    
    TUGAS UTAMA:
    Jawab pertanyaan mahasiswa berdasarkan KONTEKS DATA yang disediakan di bawah ini.
    
    ATURAN MENJAWAB:
    1. JANGAN HALUSINASI. Jika informasi tidak ada di konteks, katakan "Maaf, data tidak tersedia".
    2. JIKA DITANYA JADWAL LENGKAP: Sebutkan SEMUA mata kuliah yang ada di dalam konteks untuk kelas tersebut. Jangan disingkat.
    3. JIKA DITANYA JADWAL SPESIFIK: Hanya sebutkan mata kuliah yang ditanyakan saja.
    4. BEDAKAN JADWAL: Perhatikan baik-baik apakah konteksnya "Jadwal UTS" atau "Jadwal Kuliah/Kelas". Jangan tertukar.
    5. GAYA BAHASA: Formal, sopan, tapi luwes (seperti Customer Service profesional).
    6. FORMATTING: Gunakan Bullet Points atau Tabel markdown agar jadwal mudah dibaca.
    """

    user_prompt = f"""
    Pertanyaan User: "{user_query}"
    
    DATA KONTEKS DARI DATABASE BAAK:
    {context_text}
    
    Silakan jawab pertanyaan user berdasarkan data di atas.
    """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.3,
            max_tokens=1024
        )
        
        bot_answer = chat_completion.choices[0].message.content
        
        return ChatResponse(
            answer=bot_answer,
            sources=list(set(sources))
        )

    except Exception as e:
        print(f"❌ Error Groq API: {e}")
        return ChatResponse(
            answer="Maaf, sedang terjadi gangguan pada koneksi ke otak AI. Silakan coba sesaat lagi.",
            sources=[]
        )