import requests
import json
import time
from tqdm import tqdm

# URL API Chatbot kamu
API_URL = "http://localhost:8000/chat"

# Daftar 20 Skenario Pengetesan Lengkap
TEST_CASES = [
    # --- A. JADWAL & KALENDER (4 Soal) ---
    {"kategori": "Jadwal Kuliah", "pertanyaan": "Tampilkan jadwal kuliah kelas 4IA01"},
    {"kategori": "Jadwal UTS",    "pertanyaan": "Kapan jadwal UTS untuk kelas 4IA01?"}, 
    {"kategori": "Jadwal Dosen",  "pertanyaan": "Hari apa saja dosen MAUKAR mengajar?"},
    {"kategori": "Kalender",      "pertanyaan": "Kapan batas akhir pengisian KRS?"},
    
    # --- B. LAYANAN PERKULIAHAN & UJIAN (4 Soal) ---
    {"kategori": "Ujian Bentrok", "pertanyaan": "Bagaimana cara mengurus ujian bentrok?"},
    {"kategori": "FRS",           "pertanyaan": "Apa itu Formulir Rencana Studi?"}, # <-- TAMBAHAN 1
    {"kategori": "Cek Nilai",     "pertanyaan": "Dimana saya bisa melihat nilai ujian?"},
    {"kategori": "Daftar Ulang",  "pertanyaan": "Kapan jadwal daftar ulang mahasiswa baru?"},

    # --- C. ADMINISTRASI AKADEMIK (4 Soal) ---
    {"kategori": "Cuti Akademik", "pertanyaan": "Apa syarat mengajukan cuti akademik?"},
    {"kategori": "Pindah Jurusan","pertanyaan": "Saya ingin pindah jurusan, bagaimana prosedurnya?"},
    {"kategori": "Tidak Aktif",   "pertanyaan": "Bagaimana jika saya tidak aktif kuliah semester ini?"}, # <-- TAMBAHAN 2
    {"kategori": "Pindah Lokasi", "pertanyaan": "Cara pindah lokasi atau waktu kuliah?"}, # <-- TAMBAHAN 3
    
    # --- D. BUKU PEDOMAN & DOKUMEN (4 Soal) ---
    {"kategori": "Pedoman", "pertanyaan": "Saya butuh buku pedoman penyusunan silabus"},
    {"kategori": "Pedoman", "pertanyaan": "Minta link download buku pedoman penyusunan SAP"}, # <-- TAMBAHAN 4
    {"kategori": "Pedoman", "pertanyaan": "Ada buku pedoman tata krama mahasiswa?"},
    {"kategori": "Materi",  "pertanyaan": "Minta link download materi PPSPPT"},

    # --- E. INFO UMUM & LOKET (2 Soal) ---
    {"kategori": "Loket",   "pertanyaan": "Loket 3 BAAK melayani apa saja?"},
    {"kategori": "Loket",   "pertanyaan": "Jam berapa loket BAAK buka?"},
    
    # --- F. NEGATIVE CASE (2 Soal - Cek Keamanan/Halusinasi) ---
    {"kategori": "Negative", "pertanyaan": "Berapa harga bakso di kantin Gunadarma?"},
    {"kategori": "Negative", "pertanyaan": "Jadwal kuliah kelas 9ZZ99 (kelas ngasal)"}
]

def run_tests():
    print(f"🚀 Memulai Pengujian Otomatis ({len(TEST_CASES)} skenario)...")
    results = []

    # Menggunakan tqdm untuk loading bar
    for case in tqdm(TEST_CASES, desc="Testing Progress"):
        try:
            # Kirim request ke API
            response = requests.post(API_URL, json={"question": case["pertanyaan"]})
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer")
                sources = data.get("sources", [])
                
                # Cek sekilas apakah ada link PDF jika kategori Pedoman
                if case["kategori"] == "Pedoman" and "http" not in answer:
                    status = "⚠️ NO LINK"
                else:
                    status = "✅ OK"
            else:
                answer = f"Error Status: {response.status_code}"
                sources = []
                status = "❌ FAIL"
            
        except Exception as e:
            answer = f"Connection Error: {str(e)}"
            sources = []
            status = "❌ ERR"

        # Simpan hasil
        results.append({
            "kategori": case["kategori"],
            "tanya": case["pertanyaan"],
            "jawab": answer,
            "sumber": sources,
            "status": status
        })
        
        # Jeda 1 detik agar server tidak overload
        time.sleep(1)

    generate_report(results)

def generate_report(results):
    filename = "HASIL_TESTING_LENGKAP.md"
    print(f"\n📝 Membuat Laporan Lengkap di {filename}...")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Laporan Pengujian Bot BAAK (20 Soal)\n")
        f.write(f"Waktu Pengujian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Note: Cek file ini menggunakan Mode Preview Markdown di VS Code.\n\n")
        
        for idx, res in enumerate(results):
            icon = res['status']
            f.write(f"## {idx+1}. {res['kategori']} {icon}\n")
            f.write(f"**Q:** `{res['tanya']}`\n\n")
            f.write(f"**A:**\n{res['jawab']}\n\n")
            
            if res['sumber']:
                sumber_bersih = [s.replace('.json', '') for s in res['sumber']]
                f.write(f"> *Sumber Data: {', '.join(sumber_bersih)}*\n")
            else:
                f.write("> *Sumber Data: Tidak ditemukan*\n")
            
            f.write("\n---\n")
            
    print(f"✅ Selesai! Buka file '{filename}' untuk melihat hasil jawaban AI.")

if __name__ == "__main__":
    # Pastikan server app.py jalan di terminal lain: uvicorn app:app --reload
    try:
        # Pengecekan koneksi awal
        requests.get("http://localhost:8000/docs", timeout=3)
        run_tests()
    except:
        print("\n❌ GAGAL KONEKSI!")
        print("Pastikan server Backend sudah jalan. Ketik di terminal lain:")
        print("uvicorn app:app --reload")