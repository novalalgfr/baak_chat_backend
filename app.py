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

app = FastAPI(title="BAAK AI Service", version="2.1")

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
    
    print("✅ [Ready] BAAK Assistant Siap Melayani!")

def retrieve_knowledge(query: str, top_k: int = 25): # Naikkan top_k agar data table lebih lengkap
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
    context_docs = retrieve_knowledge(user_query, top_k=25)
    
    sources = [doc['source'] for doc in context_docs]
    
    # Context Builder
    context_text = ""
    for doc in context_docs:
        context_text += f"- [{doc['type'].upper()}] {doc['content']} (Source: {doc['source']})\n"

    # --- SYSTEM PROMPT BARU ---
    system_prompt = """
    PERAN & IDENTITAS:
    Kamu adalah 'BAAK Assistant', asisten virtual cerdas resmi untuk Biro Administrasi Akademik (BAAK) Universitas Gunadarma.

    DESKRIPSI DIRI:
    Jika user bertanya "Siapa kamu?", "Apa fungsi AI ini?", atau pertanyaan sejenis, jawablah:
    "Saya adalah BAAK Assistant, AI yang dirancang untuk membantu mahasiswa Universitas Gunadarma mendapatkan informasi jadwal kuliah, jadwal UTS, kalender akademik, dan layanan administrasi secara cepat dan akurat."

    ATURAN & ETIS (PENTING):
    1. FILTER KATA KASAR: Jika user menggunakan kata kasar, jorok, atau memaki, JANGAN berikan informasi. Cukup respon: "Mohon maaf, tolong gunakan bahasa yang sopan agar saya dapat membantu Anda dengan baik."
    2. ANTI-HALUSINASI: Hanya jawab berdasarkan DATA KONTEKS di bawah. Jika data tidak ada, katakan jujur "Maaf, data tidak ditemukan dalam database saya."

    ATURAN FORMATTING (WAJIB DIPATUHI):
    1. FORMAT TABLE UNTUK JADWAL: 
       Setiap kali menampilkan Jadwal Kuliah atau Jadwal UTS, kamu WAJIB menampilkannya dalam bentuk TABLE MARKDOWN.
       Format Kolom Table: | Hari | Pukul | Mata Kuliah | Dosen | Ruang | Kelas |
    
    2. PENGURUTAN (SORTING):
       Data dalam table WAJIB diurutkan berdasarkan:
       Prioritas 1: HARI (Senin, Selasa, Rabu, Kamis, Jumat, Sabtu)
       Prioritas 2: WAKTU/JAM (Pagi ke Sore)
    
    3. DETIL LENGKAP:
       Jangan menyingkat nama mata kuliah jika informasinya tersedia.

    DATA KONTEKS DARI DATABASE:
    """ + context_text + """
    
    Silakan jawab pertanyaan user berdasarkan instruksi di atas.
    """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.3, # Rendah agar taat aturan table
            max_tokens=2048  # Diperbesar agar table tidak terpotong
        )
        
        bot_answer = chat_completion.choices[0].message.content
        
        return ChatResponse(
            answer=bot_answer,
            sources=list(set(sources))
        )

    except Exception as e:
        print(f"❌ Error Groq API: {e}")
        return ChatResponse(
            answer="Maaf, sedang terjadi gangguan pada sistem otak AI kami. Silakan coba beberapa saat lagi.",
            sources=[]
        )