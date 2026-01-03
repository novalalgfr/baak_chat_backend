from app.core.state import resources

def retrieve_knowledge(query: str, top_k: int = 12):
    collection = resources.get('collection')
    model = resources.get('embedding_model')

    if not collection or not model:
        return []

    # BGE-M3 standard encoding
    query_vector = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )
    
    documents = []
    if results['documents']:
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            
            documents.append({
                "content": results['documents'][0][i],
                "source": meta.get('source', 'unknown'),
                "kategori": meta.get('kategori', 'Umum'),
                "topik": meta.get('topik', ''),
                "hari_sort": meta.get('hari_sort', 99),
                "waktu_sort": meta.get('waktu_sort', 9999),
                "tanggal_sort": meta.get('tanggal_sort', 0)
            })
    
    # Sorting prioritas
    documents.sort(key=lambda x: (x['tanggal_sort'], x['hari_sort'], x['waktu_sort']))
    return documents