import sqlite3
from types import SimpleNamespace
from fastapi import FastAPI, Request, Query, Body, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import asyncio
import numpy as np
import requests
import time
import json
import unicodedata
import httpx

from sentence_transformers import SentenceTransformer

# LangChain + Ollama definitivo
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from file_versioning import router as versioning_router
from file_versioning import save_file_version
from file_management import router as file_router
from memory_embeddings import embed_all_records
from memory_embeddings import search_similar_memories

print("[DEBUG] main.py caricato correttamente.")

# =====================================
# Utility: Pulizia Sicura
# =====================================
def clean_text(text: str) -> str:
    if not text:
        return ""
    try:
        text = unicodedata.normalize('NFKC', text)
        text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        text = text.replace("\u00a0", " ").replace("\xa0", " ")
        text = " ".join(text.split())
    except Exception as e:
        print(f"[Pulizia] Errore pulendo testo: {e}")
    return text

# =====================================
# Configurazione del Database
# =====================================
DB_PATH = "G:/PROGETTI/Local LLM/LobeChat.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT,
        response TEXT,
        tag TEXT,
        scope TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        embedding BLOB
    );
''')
conn.commit()
conn.close()

# =====================================
# LangChain + Ollama configurato correttamente
# =====================================
llm = OllamaLLM(
    model="localmistralinstruct",
    base_url="http://localhost:11434",
    system_message="Sei un assistente conversazionale. Rispondi in modo naturale, amichevole ma conciso, mantenendo solo le informazioni tecnicamente essenziali.",
    options={"num_predict": 50, "temperature": 0.5, "top_k": 1, "seed": 42, "repeat_last_n": 40}
)

memory_obj = SQLChatMessageHistory(
    session_id="test_session_id",
    connection_string=f"sqlite:///{DB_PATH}"
)

def get_session_history():
    msgs = memory_obj.get_messages()
    return SimpleNamespace(messages=msgs)

conversation_chain = RunnableWithMessageHistory(
    runnable=llm,
    get_session_history=get_session_history
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# Funzione salvataggio memoria
# =====================================
def save_memory(prompt, response, tag=None, scope=None):
    try:
        prompt = clean_text(prompt)
        response = clean_text(response)

        embedding_model_local = SentenceTransformer("all-MiniLM-L6-v2")
        combined_text = f"{prompt} {response}"
        embedding = embedding_model_local.encode(combined_text)
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO memories (prompt, response, tag, scope, embedding) VALUES (?, ?, ?, ?, ?)",
            (prompt, response, tag, scope, embedding_blob)
        )
        conn.commit()
        print("[DEBUG] Salvataggio in 'memories' con embedding completato.")
    except Exception as e:
        print(f"[ERRORE SALVATAGGIO MEMORIA] {e}")
    finally:
        conn.close()

    try:
        save_file_version(f"ai_response_{tag or 'default'}.txt", response)
        print(f"[Versioning] Salvata versione per file: ai_response_{tag or 'default'}.txt")
    except Exception as e:
        print(f"[Versioning] Errore durante il salvataggio versione: {e}")

# =====================================
# Inizializzazione FastAPI
# =====================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

public_path = os.path.join(os.path.dirname(__file__), "public")
if os.path.isdir(public_path):
    app.mount("/static", StaticFiles(directory=public_path, html=True), name="static")

app.include_router(versioning_router, prefix="/versioning", tags=["versioning"])
app.include_router(file_router, prefix="/file", tags=["file_management"])

# =====================================
# Endpoints
# =====================================
@app.get("/ping")
async def ping():
    return {"message": "Server is running"}

@app.post("/ricorda")
async def ricorda_dato(request: Request):
    print("POST /ricorda chiamato!")
    try:
        data = await request.json()
        msg = clean_text(data.get("messaggio", ""))
        tag = clean_text(data.get("tag", None))
        scope = clean_text(data.get("scope", None))
    except Exception as e:
        print("[ERRORE PARSING JSON]", e)
        return {"error": "Error parsing JSON"}

    try:
        simili = [r for r in search_similar_memories(msg, top_k=10) if r["similarity"] >= 0.45][:3]
        contesto = "\n".join([f"Domanda: {clean_text(r['prompt'])}\nRisposta: {clean_text(r['response'])}" for r in simili])
        prompt_contestuale = f"{contesto}\n\nDomanda: {msg}"
        prompt_contestuale = clean_text(prompt_contestuale)  # Pulizia forzata su tutto prima di inviare

        print(f"[DEBUG] Prompt contestuale costruito:\n{prompt_contestuale}")

        # === DEBUG FASE 5: Byte analysis ===
        print("[DEBUG] CONTENUTO CONCATENATO (repr):", repr(prompt_contestuale))
        try:
            encoded = prompt_contestuale.encode("utf-8")
            print("[DEBUG] BYTES UTF-8:", encoded)
        except UnicodeEncodeError as e:
            print("[ERRORE UTF-8 in encode]:", str(e))
        print("[DEBUG] SAFE BYTES:", prompt_contestuale.encode("utf-8", errors="replace"))
        # === FINE DEBUG ===

        start_time = time.time()
        result = await asyncio.to_thread(conversation_chain.invoke, {"input": prompt_contestuale})
        duration = time.time() - start_time

        print(f"[DEBUG] Tempo generazione risposta: {duration:.2f} sec")
        print(f"[DEBUG] Tipo risultato: {type(result)}")

        if isinstance(result, bytes):
            try:
                response = result.decode('utf-8', errors='ignore')
            except Exception as decode_error:
                print(f"[ERRORE DECODIFICA UTF-8] {decode_error}")
                response = str(result)
        elif isinstance(result, dict):
            response = result.get("output") or str(result)
        else:
            response = str(result)

        response = clean_text(response)

    except Exception as e:
        print(f"[ERRORE GENERAZIONE AI] {repr(e)}")
        return {"error": f"Errore nella generazione della risposta AI: {str(e)}"}

    try:
        save_memory(msg, response, tag, scope)
    except Exception as e:
        print(f"[ERRORE SALVATAGGIO MEMORIA] {e}")
        return {"error": f"Errore nel salvataggio memoria: {str(e)}"}

    return {
        "status": "ok",
        "tempo_risposta": f"{duration:.2f} secondi",
        "messaggio": response,
        "tag": tag,
        "scope": scope
    }


@app.get("/embed_all")
async def embed_all():
    embed_all_records()
    return {"status": "Embeddings creati correttamente."}

@app.post("/search_memory")
async def search_memory_endpoint(query: str = Body(..., embed=True)):
    try:
        results = search_similar_memories(clean_text(query), top_k=5)
        filtered_results = [r for r in results if r["similarity"] >= 0.65]
        if filtered_results:
            best_result = max(filtered_results, key=lambda r: r["similarity"])
            return {"status": "ok", "prompt": best_result["prompt"], "response": best_result["response"], "similarity": best_result["similarity"]}
        else:
            return {"status": "no_match", "message": "Nessuna risposta sufficientemente pertinente trovata."}
    except Exception as e:
        print(f"[ERRORE SEARCH MEMORY] {e}")
        return {"error": str(e)}

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            prompt = await websocket.receive_text()
            prompt = clean_text(prompt)
            full_response = ""
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", "http://localhost:11434/api/generate", json={"model": "localmistralinstruct", "prompt": prompt, "stream": True}) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                content = line.split('data: ')[-1]
                                parsed = json.loads(content)
                                token = parsed.get("response", "")
                                full_response += token
                                await websocket.send_text(content)
                            except Exception:
                                continue
            await websocket.send_text(json.dumps({"response": "[FINE]"}))
            save_memory(prompt, full_response, scope="default")
    except Exception as e:
        await websocket.close()
        print(f"[WebSocket] Connessione chiusa: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
