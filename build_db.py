import json
import os
import shutil
import sys
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.config import MODEL_NAME, DB_PATH, COLLECTION_NAME, DATA_DIR

HARI_MAPPING = {
    "senin": 1, "selasa": 2, "rabu": 3, "kamis": 4, "jumat": 5, "sabtu": 6, "minggu": 7,
    "jum'at": 5
}

BULAN_MAPPING = {
    "januari": "01", "februari": "02", "maret": "03", "april": "04", "mei": "05", "juni": "06",
    "juli": "07", "agustus": "08", "september": "09", "oktober": "10", "november": "11", "desember": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
}

def get_waktu_sort(waktu_str):
    try:
        if not waktu_str or waktu_str == "N/A": return 9999
        clean = waktu_str.split('-')[0].strip()
        clean = clean.replace('.', '').replace(':', '')
        if clean.isdigit():
            return int(clean)
        return 9999
    except:
        return 9999

def get_tanggal_sort(tanggal_str):
    if not tanggal_str: return 0
    clean_str = tanggal_str.lower().strip()
    try:
        parts = clean_str.split(' ')
        if len(parts) >= 3:
            day = parts[0].zfill(2)
            month = BULAN_MAPPING.get(parts[1], "00")
            year = parts[2]
            if month != "00": return int(f"{year}{month}{day}")
    except:
        pass
    return 0 

def load_data_from_json():
    print(f"[*] Reading data from: {DATA_DIR}")
    if not os.path.exists(DATA_DIR): return [], [], []

    documents, metadatas, ids = [], [], []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    
    for filename in tqdm(files, desc="Processing JSON"):
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        if isinstance(content, dict) and "kategori_utama" in content and "data" in content:
            kategori_utama = content.get("kategori_utama", "Umum")
            items = content["data"]

            if "jadwal" in kategori_utama.lower():
                jenis_jadwal = "UTS" if "uts" in kategori_utama.lower() else "Kuliah"
                
                for idx, item in enumerate(items):
                    kelas = item.get('kelas', 'N/A')
                    matkul = item.get('mata_kuliah', 'N/A')
                    hari = item.get('hari', 'N/A')
                    waktu = item.get('waktu', 'N/A')
                    ruang = item.get('ruang', 'N/A')
                    tanggal = item.get('tanggal', '')
                    dosen = item.get('dosen')
                    dosen_str = f"Dosen: {dosen}. " if dosen else "" 

                    text_content = (
                        f"Jadwal {jenis_jadwal} {('Kelas ' + kelas) if jenis_jadwal == 'UTS' else 'Reguler Kelas ' + kelas}. "
                        f"Mata Kuliah: {matkul}. {dosen_str}"
                        f"Waktu: Hari {hari}, {('Tanggal ' + tanggal + ', ') if tanggal else ''}Pukul {waktu}. "
                        f"Lokasi: Ruang {ruang}. ({kategori_utama})"
                    )

                    documents.append(text_content)
                    metadatas.append({
                        "source": filename,
                        "type": "jadwal",
                        "kategori": kategori_utama,
                        "kelas": kelas,
                        "hari_sort": HARI_MAPPING.get(hari.lower(), 99),
                        "waktu_sort": get_waktu_sort(waktu),
                        "tanggal_sort": get_tanggal_sort(tanggal)
                    })
                    ids.append(f"{filename}_{idx}")

            else:
                for idx, item in enumerate(items):
                    sub_topik = item.get('sub_topik', 'Informasi')
                    deskripsi = item.get('deskripsi', '')
                    
                    detail_text = ""
                    for key in ['poin_penting', 'syarat', 'prosedur']:
                        if key in item and item[key]:
                            isi_list = item[key]
                            if isinstance(isi_list, list):
                                detail_text += f"\n{key.replace('_',' ').title()}:\n" + "\n".join([f"- {x}" for x in isi_list])
                    
                    link_text = ""
                    links = item.get('link_terkait', [])
                    if links:
                        link_text += "\nDokumen Terkait/Download:"
                        for link in links:
                            link_text += f"\n- {link.get('judul', 'Dokumen')}: {link.get('url', '#')}"

                    text_content = f"Topik: {sub_topik} ({kategori_utama}).\nPenjelasan: {deskripsi}\n{detail_text}{link_text}"

                    documents.append(text_content)
                    metadatas.append({
                        "source": filename,
                        "type": "prosedur",
                        "kategori": kategori_utama,
                        "topik": sub_topik,
                        "hari_sort": 0, "waktu_sort": 0, "tanggal_sort": 0
                    })
                    ids.append(f"{filename}_{sub_topik}_{idx}")

    return documents, metadatas, ids

def main():
    print(f"\n🚀 STARTING BAAK AI TRAINING with {MODEL_NAME}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Running on device: {device.upper()}")

    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print("🗑️  Old database removed.")
        except Exception as e:
            print(f"Warning: Could not delete folder: {e}")

    docs, metas, ids = load_data_from_json()
    if not docs:
        print("❌ No documents found.")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.create_collection(name=COLLECTION_NAME)

    print("⏳ Loading Embedding Model...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    batch_size = 100
    print(f"📊 Generating Embeddings for {len(docs)} documents...")
    
    embeddings = model.encode(docs, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True).tolist()

    print("💾 Saving to ChromaDB...")
    for i in tqdm(range(0, len(docs), batch_size), desc="Inserting to DB"):
        collection.add(
            documents=docs[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    print("\n✅ SUCCESS! Database updated.")

if __name__ == "__main__":
    main()