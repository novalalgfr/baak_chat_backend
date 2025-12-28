import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data') 
DB_PATH = os.path.join(BASE_DIR, 'chroma_db')

COLLECTION_NAME = "baak_knowledge"
MODEL_NAME = "intfloat/multilingual-e5-small" 

def load_data_from_json():
    print(f"[*] Membaca data dari: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"[!] Folder data tidak ditemukan di: {DATA_DIR}")
        return [], [], []

    documents = []
    metadatas = []
    ids = []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    
    for filename in tqdm(files, desc="Processing JSON"):
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        if isinstance(data, dict) and ("kalender_akademik" in data or "layanan_loket" in data):
            
            if "kalender_akademik" in data:
                for idx, item in enumerate(data["kalender_akademik"]):
                    kegiatan = item.get('kegiatan', 'Kegiatan Akademik')
                    tanggal = item.get('tanggal', 'Tanggal belum ditentukan')
                    
                    text_content = f"Kalender Akademik TA 2025/2026: {kegiatan} dilaksanakan pada tanggal {tanggal}."
                    
                    documents.append(f"passage: {text_content}")
                    metadatas.append({"source": filename, "type": "kalender", "topik": kegiatan})
                    ids.append(f"kalender_{idx}")

            if "layanan_loket" in data:
                jam_layanan = [item.get('waktu') for item in data["layanan_loket"] if item.get('waktu')]
                if jam_layanan:
                    jam_str = ", ".join(jam_layanan)
                    text_content = f"Informasi Jam Operasional Layanan Loket BAAK: {jam_str}."
                    
                    documents.append(f"passage: {text_content}")
                    metadatas.append({"source": filename, "type": "loket"})
                    ids.append(f"loket_info")

        elif isinstance(data, dict) and "judul" in data:
            text_content = f"Topik: {data.get('judul', 'Info')}.\n"
            if data.get('deskripsi'):
                text_content += "Penjelasan: " + " ".join(data['deskripsi']) + "\n"
            if data.get('prosedur'):
                text_content += "Langkah-langkah: " + " ".join(data['prosedur'])
            
            documents.append(f"passage: {text_content}")
            metadatas.append({"source": filename, "type": "prosedur", "title": data['judul']})
            ids.append(filename)

        elif isinstance(data, list) and len(data) > 0 and "hari" in data[0]:
            kelas_id = filename.replace("jadwal_kuliah_", "").replace("jadwal_uts_", "").replace(".json", "")
            jenis = "UTS" if "uts" in filename else "Kuliah"

            for idx, item in enumerate(data):
                tanggal = item.get('tanggal', '') 
                hari = item.get('hari', 'N/A')
                waktu = item.get('waktu', 'N/A')
                ruang = item.get('ruang', 'N/A')
                matkul = item.get('mata_kuliah', 'N/A')
                dosen = item.get('dosen', 'N/A')
                
                text_content = (
                    f"{matkul}. "
                    f"Dosen: {dosen}. "
                    f"Kelas: {kelas_id}. "
                    f"Jadwal {jenis}: Hari {hari}, Tanggal {tanggal}, Pukul {waktu}, di Ruang {ruang}."
                )
                
                documents.append(f"passage: {text_content}")
                metadatas.append({"source": filename, "type": "jadwal", "kelas": kelas_id})
                ids.append(f"{filename}_{idx}")

        elif isinstance(data, list) and len(data) > 0 and ("link_pdf" in data[0] or "url" in data[0]):
            for idx, item in enumerate(data):
                judul = item.get('judul') or item.get('jurusan') or item.get('nama') or "Dokumen"
                link = item.get('link_pdf') or item.get('url') or "#"
                text_content = f"Tersedia dokumen PDF tentang '{judul}'. Link download: {link}"
                documents.append(f"passage: {text_content}")
                metadatas.append({"source": filename, "type": "dokumen"})
                ids.append(f"{filename}_{idx}")

    return documents, metadatas, ids

def main():
    print("\n🚀 MEMULAI PROSES TRAINING AI BAAK (FULL RE-BUILD)...")
    docs, metas, ids = load_data_from_json()
    
    if not docs: 
        print("❌ Tidak ada data yang ditemukan. Pastikan folder ../data berisi file JSON.")
        return

    print(f"\n[*] Inisialisasi ChromaDB di: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)

    print(f"[*] Loading Model Embedding ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[*] Embedding {len(docs)} data ke dalam Vector Database...")
    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    print("\n✅ SUKSES! Database BAAK sudah diperbarui.")
    print("   Sekarang AI sudah tahu tentang Kalender Akademik, Jadwal, dan Layanan Loket.")

if __name__ == "__main__":
    main()
