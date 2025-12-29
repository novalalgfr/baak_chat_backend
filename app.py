import os
import chromadb
from fastapi import FastAPI
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

app = FastAPI(title="BAAK AI Service", version="2.4-Stable")

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
    
    print("⏳ [Init] Loading Resources...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    if os.path.exists(DB_PATH):
        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            vector_db_collection = client.get_collection(COLLECTION_NAME)
            print("✅ Connected to ChromaDB.")
        except:
            print("⚠️ Collection not found. Run 'build_db.py' first.")
    else:
        print("⚠️ Database path missing. Run 'build_db.py' first.")
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ BAAK Assistant Ready!")

def retrieve_knowledge(query: str, top_k: int = 25):
    # Note: top_k dikurangi sedikit (30 -> 25) untuk mengurangi noise jadwal lain (Masalah #1)
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
                "type": meta.get('type', 'unknown'),
                "hari_sort": meta.get('hari_sort', 99),
                "waktu_sort": meta.get('waktu_sort', 99),
                "tanggal_sort": meta.get('tanggal_sort', 0)
            })
    
    # Sorting Logic (Tetap dipertahankan karena sudah bagus)
    documents.sort(key=lambda x: (x['tanggal_sort'], x['hari_sort'], x['waktu_sort']))
    
    return documents

@app.post("/chat", response_model=ChatResponse)
def chat_with_baak(request: ChatRequest):
    user_query = request.question
    print(f"\nUser Query: {user_query}")
    
    context_docs = retrieve_knowledge(user_query, top_k=25)
    sources = [doc['source'] for doc in context_docs]
    
    if not context_docs:
        return ChatResponse(
            answer="Mohon maaf, saya tidak menemukan informasi terkait di database saat ini.",
            sources=[]
        )

    context_text = ""
    for doc in context_docs:
        context_text += f"- {doc['content']} (Sumber: {doc['source']})\n"

    system_prompt = """
    Kamu adalah 'BAAK Assistant', AI resmi BAAK Universitas Gunadarma.

    TUGAS:
    Jawab pertanyaan mahasiswa dengan TEPAT berdasarkan DATA KONTEKS yang diberikan.

    ATURAN LOGIKA (WAJIB DIPATUHI):
    
    1. **PEMISAHAN JADWAL:**
       - Jika user bertanya "Jadwal Kuliah" (Reguler), HANYA ambil data jadwal kelas biasa. JANGAN masukan data UTS/Ujian kecuali diminta.
       - Jika user bertanya "Jadwal UTS", HANYA ambil data ujian.
       - Jika data jadwal bercampur di konteks, filterlah secara cerdas.
    
    2. **PROSEDUR & ADMINISTRASI:**
       - Jika pertanyaan tentang "Cara", "Syarat", "Prosedur", atau "Jam Buka", rangkum teks narasi menjadi FORMAT LIST (Bullet Points) atau TABEL sederhana agar mudah dibaca.
       - Jangan menolak menjawab jika datanya berbentuk teks narasi (bukan tabel). Baca dan pahami isinya.
    
    3. **NEGOSIASI DATA:**
       - Jika user mencari X (misal: "Lihat Nilai"), tapi data yang ada hanya Y (misal: "Komplain Nilai"), JANGAN jawab "Tidak Tahu".
       - Jawab: "Maaf, info spesifik [X] tidak ada, tapi saya punya info [Y] yang mungkin membantu: ..." lalu jelaskan [Y].

    4. **LINK DOWNLOAD:**
       - Jika di konteks ada URL/Link PDF, WAJIB ditampilkan.
       - Format: [Nama Dokumen](URL).

    FORMATTING OUTPUT:
    - **Jadwal/Jam Buka:** Gunakan TABLE MARKDOWN.
      | Hari | Pukul | Kegiatan/Matkul | Ruang/Keterangan |
    - **Prosedur:** Gunakan Bullet Points.
    - **Gaya Bahasa:** Formal, Ramah, Solutif. Jika ada kata kasar dari user, tegur dengan sopan.
    
    DESKRIPSI DIRI:
    Jika ditanya "Tentang apa ai ini?", jawab bahwa kamu adalah asisten untuk informasi jadwal dan administrasi akademik BAAK.

    DATA KONTEKS (SUDAH TERURUT SECARA KRONOLOGIS):
    """ + context_text

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="openai/gpt-oss-120b", 
            temperature=0.3,
            max_tokens=2048
        )
        
        bot_answer = chat_completion.choices[0].message.content
        
        return ChatResponse(
            answer=bot_answer,
            sources=list(set(sources))
        )

    except Exception as e:
        print(f"Error Groq: {e}")
        return ChatResponse(
            answer="Maaf, sedang terjadi gangguan koneksi ke otak AI.",
            sources=[]
        )