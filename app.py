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

app = FastAPI(title="BAAK AI Service", version="3.5-Ultimate")

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

def retrieve_knowledge(query: str, top_k: int = 30):
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
            raw_content = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            
            clean_content = raw_content.replace("passage: ", "")
            
            documents.append({
                "content": clean_content,
                "source": meta.get('source', 'unknown'),
                "type": meta.get('type', 'general'),
                "kategori": meta.get('kategori', 'Umum'),
                "topik": meta.get('topik', ''),
                "hari_sort": meta.get('hari_sort', 99),
                "waktu_sort": meta.get('waktu_sort', 9999),
                "tanggal_sort": meta.get('tanggal_sort', 0)
            })
    
    documents.sort(key=lambda x: (x['tanggal_sort'], x['hari_sort'], x['waktu_sort']))
    
    return documents

@app.post("/chat", response_model=ChatResponse)
def chat_with_baak(request: ChatRequest):
    user_query = request.question
    print(f"\nUser Query: {user_query}")
    
    context_docs = retrieve_knowledge(user_query, top_k=30)
    sources = list(set([doc['source'] for doc in context_docs]))
    
    if not context_docs:
        return ChatResponse(
            answer="Mohon maaf, saya tidak menemukan informasi terkait di database saat ini.",
            sources=[]
        )

    context_text = ""
    for doc in context_docs:
        label = doc['kategori']
        if doc['topik']:
            label += f" - {doc['topik']}"
            
        context_text += f"[{label}]\n{doc['content']}\n\n"

    system_prompt = """
    PERAN:
    Kamu adalah 'BAAK Assistant', AI resmi BAAK Universitas Gunadarma.
    Tugasmu adalah menjawab pertanyaan mahasiswa berdasarkan DATA KONTEKS yang diberikan.

    ATURAN LOGIKA (WAJIB DIPATUHI):
    1. **PEMISAHAN JADWAL:**
       - Jika user bertanya "Jadwal Kuliah" (Reguler), HANYA ambil data jadwal kelas biasa. JANGAN masukan data UTS/Ujian kecuali diminta.
       - Jika user bertanya "Jadwal UTS", HANYA ambil data ujian yang ada Tanggal-nya.
       - Jika data jadwal bercampur di konteks, filterlah sesuai pertanyaan user.
    
    2. **PROSEDUR & ADMINISTRASI:**
       - Jika pertanyaan tentang "Cara", "Syarat", "Prosedur", rangkum teks narasi menjadi FORMAT LIST (Bullet Points) atau TABEL sederhana agar mudah dibaca.
       - Baca detail prosedur dari data konteks dengan teliti.
    
    3. **NEGOSIASI DATA:**
       - Jika user mencari X tapi hanya ada Y, JANGAN jawab "Tidak Tahu".
       - Jawab: "Maaf, info spesifik [X] tidak ada, tapi saya punya info [Y] yang mungkin membantu: ..." lalu jelaskan [Y].

    ATURAN FORMATTING & TATA LETAK (STRICT):
    1. **STRUKTUR JAWABAN:**
       - Gunakan **Heading 3 (###)** untuk memisahkan bagian (Contoh: ### Prosedur, ### Syarat, ### Download).
       - Gunakan Bullet Points (-) untuk rincian.

    2. **JIKA JADWAL KULIAH REGULER:**
       - Format: | Hari | Pukul | Mata Kuliah | Dosen | Ruang | Kelas |
       - **DILARANG** membuat kolom 'Tanggal' untuk kuliah reguler.
    
    3. **JIKA JADWAL UTS / UJIAN:**
       - Format: | Hari | Tanggal | Pukul | Mata Kuliah | Dosen | Ruang | Kelas |
       - Wajib ada kolom 'Tanggal'.

    4. **ATURAN URUTAN (SORTING):**
       - Data konteks yang saya berikan SUDAH TERURUT (Pagi -> Sore).
       - Kamu **DILARANG MENGUBAH URUTAN BARIS** dalam tabel. Salin persis urutan baris dari data konteks.

    5. **DOKUMEN & LINK (CRITICAL):**
       - Jika di data konteks ada URL/Link PDF, kamu WAJIB menampilkannya.
       - Format: - [Nama Dokumen](URL_LINK)
       - **ANTI-BROKEN LINK:** Jika URL mengandung spasi, kamu WAJIB menggantinya dengan %20 (Contoh: '.../Administrasi Akademik/...' menjadi '.../Administrasi%20Akademik/...').
       - Jangan menyuruh cek website jika link sudah tersedia di konteks.

    DATA KONTEKS (SUDAH TERURUT):
    """ + context_text

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3, 
            max_tokens=2048
        )
        
        bot_answer = chat_completion.choices[0].message.content
        
        return ChatResponse(
            answer=bot_answer,
            sources=sources
        )

    except Exception as e:
        print(f"Error Groq: {e}")
        return ChatResponse(
            answer="Maaf, sedang terjadi gangguan koneksi ke server AI.",
            sources=[]
        )