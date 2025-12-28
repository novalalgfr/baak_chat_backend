# BAAK Chat Backend (RAG API) 🧠

Backend service yang mengimplementasikan sistem RAG (Retrieval-Augmented Generation). Berfungsi untuk mencari data relevan di database vektor dan menghasilkan jawaban melalui LLM Groq.

## 🛠️ Tech Stack

- **FastAPI** - Framework API modern dan cepat
- **ChromaDB** - Vector Database untuk penyimpanan embeddings
- **Sentence-Transformers** - Embedding Model (multilingual-e5-small)
- **Groq API** - Large Language Model (Llama-3.3-70b)
- **Python-dotenv** - Manajemen environment variables

## ⚙️ Persiapan (Setup)

### 1. Masuk ke Direktori Project

```bash
cd baak_chat_backend
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Environment

Buat file `.env` di root directory dan masukkan API Key Groq:

```env
GROQ_API_KEY=gsk_xxx...
```

### 4. Build Database Vektor

Latih (Build) database vektor dari hasil scraping:

```bash
python build_db.py
```

## 🚀 Menjalankan Server

```bash
uvicorn app:app --reload --port 8000
```

API akan tersedia di `http://localhost:8000`

## 📚 Dokumentasi API

Akses dokumentasi interaktif Swagger UI di:

```
http://localhost:8000/docs
```

## 📁 Struktur Project

```
baak_chat_backend/
├── app.py              # Entry point FastAPI
├── build_db.py         # Script build vector database
├── requirements.txt    # Daftar dependensi
├── .env               # Environment variables (tidak di-commit)
├── chroma_db/         # Folder ChromaDB storage
└── README.md          # Dokumentasi project
```

## 🔄 Alur Kerja RAG

1. **User Query** → Diterima oleh FastAPI endpoint
2. **Embedding** → Query diubah menjadi vector menggunakan multilingual-e5-small
3. **Retrieval** → Mencari dokumen relevan di ChromaDB
4. **Generation** → LLM Groq menghasilkan jawaban berdasarkan konteks
5. **Response** → Jawaban dikirim kembali ke client

## 🔒 Security

- Jangan commit file `.env` ke repository
- Simpan API keys dengan aman
- Gunakan environment variables untuk konfigurasi sensitif

## 📝 License

Project ini dibuat untuk keperluan akademik Universitas Gunadarma.

---

**Note:** Pastikan API Key Groq valid dan memiliki quota yang cukup sebelum menjalankan service.
