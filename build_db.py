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

BULAN_MAPPING = {
    "januari": "01", "februari": "02", "maret": "03", "april": "04", "mei": "05", "juni": "06",
    "juli": "07", "agustus": "08", "september": "09", "oktober": "10", "november": "11", "desember": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
}

def get_waktu_sort(waktu_str):
    try:
        if not waktu_str or waktu_str == "N/A": return 99
        first_digit = waktu_str.split('/')[0].strip().replace('.', '')
        return int(first_digit)
    except:
        return 99

def get_tanggal_sort(tanggal_str):
    """Mengubah format '8 Mei 2025' atau '08-05-2025' menjadi integer 20250508"""
    if not tanggal_str or tanggal_str.lower() in ["", "n/a", "tanggal belum ditentukan"]:
        return 0

    clean_str = tanggal_str.lower().strip()
    
    # Coba Format: DD-MM-YYYY / DD/MM/YYYY
    try:
        clean_str = clean_str.replace('/', '-')
        parts = clean_str.split('-')
        if len(parts) == 3:
            return int(f"{parts[2]}{parts[1]}{parts[0]}")
    except:
        pass

    # Coba Format: DD Bulan YYYY
    try:
        parts = clean_str.split(' ')
        if len(parts) >= 3:
            day = parts[0].zfill(2)
            month_str = parts[1]
            year = parts[2]
            month = BULAN_MAPPING.get(month_str, "00")
            if month != "00":
                return int(f"{year}{month}{day}")
    except:
        pass
        
    return 0 

def load_data_from_json():
    print(f"[*] Reading data from: {DATA_DIR}")
    
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

        # --- TIPE 1: JADWAL (KULIAH / UTS) ---
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

                # Sorting Metadata Calculation
                hari_score = HARI_MAPPING.get(hari.lower(), 99)
                waktu_score = get_waktu_sort(waktu)
                tanggal_score = get_tanggal_sort(tanggal)

                if jenis_jadwal == "UTS":
                    text_content = (
                        f"Jadwal {jenis_jadwal} untuk Kelas {kelas_spesifik} (dan seluruh kelas awalan {prefix_kelas}). "
                        f"Mata Kuliah: {matkul}. Dosen: {dosen}. "
                        f"Waktu: Hari {hari}, Tanggal {tanggal}, Pukul {waktu}. "
                        f"Lokasi: Ruang {ruang}. "
                        f"Catatan: Jadwal UTS {prefix_kelas}01 berlaku sama untuk angkatan {prefix_kelas}."
                    )
                else:
                    text_content = (
                        f"Jadwal {jenis_jadwal} Reguler untuk Kelas {kelas_spesifik}. "
                        f"Mata Kuliah: {matkul}. Dosen: {dosen}. "
                        f"Waktu: Hari {hari}, Pukul {waktu}. "
                        f"Lokasi: Ruang {ruang}."
                    )
                
                documents.append(f"passage: {text_content}")
                
                metadatas.append({
                    "source": filename, 
                    "type": "jadwal", 
                    "kategori": jenis_jadwal,
                    "kelas_spesifik": kelas_spesifik,
                    "kelas_prefix": prefix_kelas,
                    "dosen": dosen,
                    "matkul": matkul,
                    "hari_sort": hari_score,
                    "waktu_sort": waktu_score,
                    "tanggal_sort": tanggal_score
                })
                ids.append(f"{filename}_{idx}")

        # --- TIPE 2: KALENDER & LOKET ---
        elif isinstance(data, dict) and ("kalender_akademik" in data or "layanan_loket" in data):
            if "kalender_akademik" in data:
                for idx, item in enumerate(data["kalender_akademik"]):
                    kegiatan = item.get('kegiatan', 'Kegiatan Akademik')
                    tanggal = item.get('tanggal', 'Tanggal belum ditentukan')
                    text_content = f"Kalender Akademik: {kegiatan} dilaksanakan pada tanggal {tanggal}."
                    
                    documents.append(f"passage: {text_content}")
                    metadatas.append({
                        "source": filename, "type": "kalender", "topik": kegiatan, 
                        "hari_sort": 0, "waktu_sort": 0, "tanggal_sort": 0
                    })
                    ids.append(f"kalender_{idx}")

            if "layanan_loket" in data:
                jam_layanan = [item.get('waktu') for item in data["layanan_loket"] if item.get('waktu')]
                if jam_layanan:
                    jam_str = ", ".join(jam_layanan)
                    text_content = f"Informasi Jam Operasional Layanan Loket BAAK: {jam_str}."
                    
                    documents.append(f"passage: {text_content}")
                    metadatas.append({
                        "source": filename, "type": "loket", 
                        "hari_sort": 0, "waktu_sort": 0, "tanggal_sort": 0
                    })
                    ids.append("loket_info")

        # --- TIPE 3: PDF / LINKS ---
        elif isinstance(data, list) and len(data) > 0 and ("link_pdf" in data[0] or "url" in data[0]):
            for idx, item in enumerate(data):
                judul = item.get('judul') or item.get('jurusan') or item.get('nama') or "Dokumen"
                link = item.get('link_pdf') or item.get('url') or "#"
                text_content = f"Tersedia dokumen PDF tentang '{judul}'. Link download: {link}"
                
                documents.append(f"passage: {text_content}")
                metadatas.append({
                    "source": filename, "type": "dokumen", 
                    "hari_sort": 0, "waktu_sort": 0, "tanggal_sort": 0
                })
                ids.append(f"{filename}_{idx}")

        # --- TIPE 4: GENERAL INFO / PROSEDUR ---
        else:
            items_to_process = []
            if isinstance(data, dict) and "judul" in data:
                items_to_process = [data]
            elif isinstance(data, list) and len(data) > 0 and "judul" in data[0]:
                items_to_process = data

            if items_to_process:
                for idx, item in enumerate(items_to_process):
                    judul = item.get('judul', 'Informasi Umum')
                    
                    deskripsi = ""
                    if item.get('deskripsi'):
                        if isinstance(item['deskripsi'], list):
                            deskripsi = " ".join([str(d) for d in item['deskripsi']])
                        else:
                            deskripsi = str(item['deskripsi'])
                            
                    prosedur = ""
                    if item.get('prosedur'):
                        if isinstance(item['prosedur'], list):
                            prosedur = " ".join([str(p) for p in item['prosedur']])
                        else:
                            prosedur = str(item['prosedur'])

                    text_content = (
                        f"Topik Panduan: {judul}.\n"
                        f"Penjelasan: {deskripsi}\n"
                        f"Langkah/Prosedur: {prosedur}"
                    )

                    documents.append(f"passage: {text_content}")
                    metadatas.append({
                        "source": filename, "type": "prosedur", "title": judul,
                        "hari_sort": 0, "waktu_sort": 0, "tanggal_sort": 0
                    })
                    ids.append(f"{filename}_topik_{idx}")

    return documents, metadatas, ids

def main():
    print("\n🚀 STARTING BAAK AI TRAINING...")
    docs, metas, ids = load_data_from_json()
    
    if not docs:
        print("❌ No documents found.")
        return

    print(f"\n[*] Initializing ChromaDB at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print("[*] Loading Embedding Model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("[*] Generating Embeddings...")
    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    batch_size = 100
    print("[*] Inserting into Database...")
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    print("\n✅ SUCCESS! Database updated with Date Sorting logic.")

if __name__ == "__main__":
    main()