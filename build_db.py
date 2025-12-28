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

HARI_MAPPING = {
    "senin": 1, "selasa": 2, "rabu": 3, "kamis": 4, "jumat": 5, "sabtu": 6, "minggu": 7
}

def get_waktu_sort(waktu_str):
    try:
        first_digit = waktu_str.split('/')[0].strip()
        return int(first_digit)
    except:
        return 99

def load_data_from_json():
    print(f"[*] Membaca data dari: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
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
        except:
            continue

        if isinstance(data, list) and len(data) > 0 and "hari" in data[0]:
            jenis_jadwal = "UTS" if "uts" in filename.lower() else "Kuliah"
            default_kelas_id = filename.replace("jadwal_kuliah_", "").replace("jadwal_uts_", "").replace(".json", "")

            for idx, item in enumerate(data):
                kelas_spesifik = item.get('kelas', default_kelas_id)
                prefix_kelas = kelas_spesifik[:3] if len(kelas_spesifik) >= 3 else kelas_spesifik

                tanggal = item.get('tanggal', '') 
                hari = item.get('hari', 'N/A')
                waktu = item.get('waktu', 'N/A')
                ruang = item.get('ruang', 'N/A')
                matkul = item.get('mata_kuliah', 'N/A')
                dosen = item.get('dosen', 'N/A')

                if jenis_jadwal == "UTS":
                    text_content = (
                        f"Jadwal {jenis_jadwal} untuk Kelas {kelas_spesifik} (dan seluruh kelas awalan {prefix_kelas}). "
                        f"Mata Kuliah: {matkul}. "
                        f"Dosen: {dosen}. "
                        f"Waktu: Hari {hari}, Tanggal {tanggal}, Pukul {waktu}. "
                        f"Lokasi: Ruang {ruang}. "
                        f"Catatan: Jadwal UTS {prefix_kelas}01 berlaku sama untuk semua kelas awalan {prefix_kelas}."
                    )
                else:
                    text_content = (
                        f"Jadwal {jenis_jadwal} Reguler untuk Kelas {kelas_spesifik}. "
                        f"Mata Kuliah: {matkul}. "
                        f"Dosen: {dosen}. "
                        f"Waktu: Hari {hari}, Pukul {waktu}. "
                        f"Lokasi: Ruang {ruang}."
                    )
                
                documents.append(f"passage: {text_content}")
                
                hari_score = HARI_MAPPING.get(hari.lower(), 99)
                waktu_score = get_waktu_sort(waktu)

                metadatas.append({
                    "source": filename, 
                    "type": "jadwal", 
                    "kategori": jenis_jadwal,
                    "kelas_spesifik": kelas_spesifik,
                    "kelas_prefix": prefix_kelas,
                    "dosen": dosen,
                    "matkul": matkul,
                    "hari_sort": hari_score,
                    "waktu_sort": waktu_score
                })
                ids.append(f"{filename}_{idx}")

        elif isinstance(data, dict) and ("kalender_akademik" in data or "layanan_loket" in data):
            if "kalender_akademik" in data:
                for idx, item in enumerate(data["kalender_akademik"]):
                    kegiatan = item.get('kegiatan', 'Kegiatan Akademik')
                    tanggal = item.get('tanggal', 'Tanggal belum ditentukan')
                    text_content = f"Kalender Akademik TA 2025/2026: {kegiatan} dilaksanakan pada tanggal {tanggal}."
                    documents.append(f"passage: {text_content}")
                    metadatas.append({
                        "source": filename,
                        "type": "kalender",
                        "topik": kegiatan,
                        "hari_sort": 0,
                        "waktu_sort": 0
                    })
                    ids.append(f"kalender_{idx}")

            if "layanan_loket" in data:
                jam_layanan = [item.get('waktu') for item in data["layanan_loket"] if item.get('waktu')]
                if jam_layanan:
                    jam_str = ", ".join(jam_layanan)
                    text_content = f"Informasi Jam Operasional Layanan Loket BAAK: {jam_str}."
                    documents.append(f"passage: {text_content}")
                    metadatas.append({
                        "source": filename,
                        "type": "loket",
                        "hari_sort": 0,
                        "waktu_sort": 0
                    })
                    ids.append("loket_info")

        elif isinstance(data, dict) and "judul" in data:
            text_content = f"Topik: {data.get('judul', 'Info')}.\n"
            if data.get('deskripsi'):
                text_content += "Penjelasan: " + " ".join(data['deskripsi']) + "\n"
            if data.get('prosedur'):
                text_content += "Langkah-langkah: " + " ".join(data['prosedur'])

            documents.append(f"passage: {text_content}")
            metadatas.append({
                "source": filename,
                "type": "prosedur",
                "title": data['judul'],
                "hari_sort": 0,
                "waktu_sort": 0
            })
            ids.append(filename)

        elif isinstance(data, list) and len(data) > 0 and ("link_pdf" in data[0] or "url" in data[0]):
            for idx, item in enumerate(data):
                judul = item.get('judul') or item.get('jurusan') or item.get('nama') or "Dokumen"
                link = item.get('link_pdf') or item.get('url') or "#"
                text_content = f"Tersedia dokumen PDF tentang '{judul}'. Link download: {link}"
                documents.append(f"passage: {text_content}")
                metadatas.append({
                    "source": filename,
                    "type": "dokumen",
                    "hari_sort": 0,
                    "waktu_sort": 0
                })
                ids.append(f"{filename}_{idx}")

    return documents, metadatas, ids

def main():
    print("\n🚀 MEMULAI PROSES TRAINING AI BAAK (FULL RE-BUILD V2)...")
    docs, metas, ids = load_data_from_json()
    
    if not docs:
        return

    print(f"\n[*] Inisialisasi ChromaDB di: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print("[*] Loading Model...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    print("\n✅ SUKSES! Database diperbarui dengan Logika Sorting & Pemisahan Konteks.")

if __name__ == "__main__":
    main()
