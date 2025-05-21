import sqlite3
import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer
import httpx

# =====================================
# Configurazione Tokenizer e Chunking
# =====================================
# Usa l'encoding cl100k_base (compatibile con Mistral Q5)
encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Restituisce il numero di token nel testo, secondo l'encoding scelto.
    """
    return len(encoding.encode(text))


def chunk_text(text: str, max_tokens: int = 1500) -> list[str]:
    """
    Divide il testo in blocchi che non superano max_tokens token.
    """
    blocks = text.split("\n\n")
    chunks = []
    current = ""
    for block in blocks:
        # Se un singolo block è troppo grande, spezzalo per riga
        if count_tokens(block) > max_tokens:
            for line in block.split("\n"):
                if count_tokens(line) > max_tokens:
                    # Tronca alla massima lunghezza in token
                    tokens = encoding.encode(line)[:max_tokens]
                    truncated = encoding.decode(tokens)
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(truncated)
                else:
                    if count_tokens(current + "\n" + line) > max_tokens:
                        if current:
                            chunks.append(current)
                        current = line
                    else:
                        current += ("\n" if current else "") + line
            continue
        # Prova ad accodare il block
        if count_tokens(current + "\n\n" + block) > max_tokens:
            if current:
                chunks.append(current)
            current = block
        else:
            current += ("\n\n" if current else "") + block
    if current:
        chunks.append(current)
    return chunks


# =====================================
# Index Vettoriale FAISS
# =====================================
EMBED_DIM = 384
INDEX_PATH = "rag_index.faiss"
ID_MAP_PATH = "rag_id_map.npy"

# Carica o ricostruisce l'indice FAISS da SQLite

def load_faiss_index(db_path=INDEX_PATH, id_map_path=ID_MAP_PATH):
    try:
        index = faiss.read_index(db_path)
        id_map = np.load(id_map_path)
    except Exception:
        conn = sqlite3.connect("LobeChat.db")
        cur = conn.cursor()
        cur.execute("SELECT id, embedding FROM memories ORDER BY id ASC")
        rows = cur.fetchall()
        conn.close()
        embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
        id_map = np.array([row[0] for row in rows])
        index = faiss.IndexFlatL2(EMBED_DIM)
        index.add(np.vstack(embeddings))
        faiss.write_index(index, db_path)
        np.save(id_map_path, id_map)
    return index, id_map

# Inizializza globalmente:
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index, id_map = load_faiss_index()


def retrieve_similar_snippets(query: str, k: int = 5) -> list[tuple[str,str]]:
    """
    Dato un query text, ritorna i top-k (prompt, response) dalla tabella memories.
    """
    query_vec = embedding_model.encode(query)
    D, I = faiss_index.search(np.array([query_vec]), k)
    ids = [int(id_map[i]) for i in I[0] if i < len(id_map)]
    if not ids:
        return []
    conn = sqlite3.connect("LobeChat.db")
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in ids)
    cur.execute(f"SELECT prompt, response FROM memories WHERE id IN ({placeholders})", ids)
    snippets = cur.fetchall()
    conn.close()
    return snippets


def retrieve_history(chat_id: str, limit: int = 20) -> list[tuple[str,str]]:
    """
    Ritorna gli ultimi 'limit' scambi (prompt,response) per una chat.
    """
    conn = sqlite3.connect("LobeChat.db")
    cur = conn.cursor()
    cur.execute("SELECT prompt, response FROM memories WHERE scope = ? ORDER BY id DESC LIMIT ?", (chat_id, limit))
    rows = cur.fetchall()[::-1]
    conn.close()
    return rows


# =====================================
# Summarization e Prompt Builder
# =====================================
async def summarize_text(text: str, max_words: int = 200) -> str:
    summary_prompt = (
        f"### Riassumi in massimo {max_words} parole il seguente testo:\n"
        f"{text}"
    )
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "localmistralinstruct",
                "prompt": summary_prompt,
                "max_tokens": 256,
                "stream": False
            }
        )
    return resp.json().get("response", "").strip()

async def build_rag_prompt(raw_chunk: str, chat_id: str) -> str:
    """
    Costruisce il prompt per un singolo chunk di codice:
    1) Fornisce istruzioni vincolanti: modifica solo questo chunk.
    2) Invia il chunk puro senza altri contesti.
    """

    instruction = (
        "### ISTRUZIONI:\n"
        "- Sei un codice-editor esperto.\n"
        "- Quando ti invio un chunk di codice, lo riproduci esattamente, "
        "modificando solo le righe di quel chunk per migliorarle.\n"
        "- Non aggiungere né rimuovere righe fuori da questo chunk.\n"
        "- Restituisci solo il codice ottimizzato, senza spiegazioni."
    )

    parts = [
        instruction,
        "",
        "### CHUNK DI CODICE DA OTTIMIZZARE:",
        raw_chunk
    ]

    return "\n".join(parts)


# =====================================
# Funzione di chiamata LLM per diff
# =====================================
async def call_model_and_get_diff(prompt: str) -> str:
    """
    Invia il prompt al modello in modalità non-stream e ritorna tutta la risposta.
    """
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": "localmistralinstruct", "prompt": prompt, "stream": False}
        )
    return resp.json().get("response", "")
