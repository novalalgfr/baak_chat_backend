import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Gunakan PROJECT_ROOT untuk path lainnya
DB_PATH = os.path.join(PROJECT_ROOT, 'chroma_db')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')  # <-- Sekarang aman karena PROJECT_ROOT sudah ada

# Constants
COLLECTION_NAME = "baak_knowledge"
MODEL_NAME = "BAAI/bge-m3"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")