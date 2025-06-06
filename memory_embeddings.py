
import sqlite3
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# === CONFIGURAZIONE ===
DATABASE_PATH = 'G:/PROGETTI/Local LLM/LobeChat.db'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Carica il modello locale solo una volta
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === FUNZIONI PRINCIPALI ===

def generate_embedding(text: str) -> List[float]:
    """Genera un embedding per il testo usando il modello locale SentenceTransformer."""
    embedding = embedding_model.encode(text).tolist()
    return embedding

def save_embedding(record_id: int, embedding: List[float]) -> None:
    """Salva l'embedding come BLOB binario (non JSON)."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
    cursor.execute(
        "UPDATE memories SET embedding = ? WHERE id = ?",
        (embedding_blob, record_id)
    )
    conn.commit()
    conn.close()

def embed_all_records() -> None:
    """Genera e salva embeddings per tutti i record senza embedding."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, prompt, response FROM memories WHERE embedding IS NULL")
    rows = cursor.fetchall()

    for record_id, prompt, response in rows:
        full_text = f"{prompt} {response}"
        embedding = generate_embedding(full_text)
        save_embedding(record_id, embedding)
        print(f"✅ Embedding salvato per ID {record_id}")

    conn.close()

def search_similar_memories(query_text: str, top_k: int = 5) -> List[Dict]:
    """Trova i record più simili dato un testo di query."""
    query_embedding = generate_embedding(query_text)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, prompt, response, embedding FROM memories WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    similarities = []
    for record_id, prompt, response, embedding_blob in rows:
        try:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = cosine_similarity(query_embedding, embedding)
            print(f"[SIMILARITY CHECK] ID {record_id} | Similarity: {similarity:.4f} | Prompt: {prompt[:50]}...")
            similarities.append({
                "id": record_id,
                "prompt": prompt,
                "response": response,
                "similarity": similarity
            })
        except Exception as e:
            print(f"[ERRORE PARSING EMBEDDING ID {record_id}]: {e}")

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calcola la similarità coseno tra due vettori."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
