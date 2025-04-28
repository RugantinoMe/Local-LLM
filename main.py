import sqlite3
from types import SimpleNamespace
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import asyncio
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import time

# LangChain reintrodotto con profilo definitivo di produzione
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# Importa il router e la funzione di versioning
from file_versioning import router as versioning_router
from file_versioning import save_file_version

print("[DEBUG] main.py caricato correttamente.")

# =====================================
# Configurazione del Database
# =====================================
DB_PATH = "G:/PROGETTI/Local LLM/LobeChat.db"
global_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
global_cursor = global_conn.cursor()
global_cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT,
        response TEXT,
        tag TEXT,
        scope TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
''')
global_conn.commit()
global_conn.close()

# =====================================
# LangChain + Ollama (Profilo definitivo di produzione)
# =====================================
llm = OllamaLLM(
    model="mistral:latest",
    options={
        "num_predict": 150,
        "temperature": 0.1,
        "top_k": 1,
        "seed": 42,
        "repeat_last_n": 40
    }
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

# =====================================
# Warmup iniziale tecnico per attivazione GPU
# =====================================
try:
    print("[WARMUP] Avvio warmup iniziale del modello...")
    llm.invoke("Simula una discussione tecnica complessa tra due sviluppatori sul refactoring del codice legacy in un sistema distribuito su larga scala.")
    print("[WARMUP] Warmup completato.")
except Exception as e:
    print(f"[WARMUP] Errore durante il warmup: {e}")

# =====================================
# Funzione salvataggio
# =====================================
def save_memory(prompt, response, tag=None, scope=None):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO memories (prompt, response, tag, scope) VALUES (?, ?, ?, ?)",
            (prompt, response, tag, scope)
        )
        conn.commit()
        print("[DEBUG] Salvataggio in 'memories' completato.")
    finally:
        conn.close()

    file_name = f"ai_response_{tag or 'default'}.txt"
    try:
        save_file_version(file_name, response)
        print(f"[Versioning] Salvata versione per file: {file_name}")
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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
app.include_router(versioning_router, prefix="/versioning", tags=["versioning"])

# =====================================
# Endpoint di base
# =====================================
@app.get("/ping")
async def ping():
    return {"message": "Server is running"}

@app.post("/test")
async def test_endpoint(request: Request):
    try:
        data = await request.json()
    except Exception:
        return {"error": "Error parsing JSON"}
    return {"message": "Test endpoint reached", "data": data}

# =====================================
# Endpoint /ricorda (con profiling dettagliato)
# =====================================
@app.post("/ricorda")
async def ricorda_dato(request: Request):
    print("POST /ricorda chiamato!")
    try:
        data = await request.json()
        msg = data.get("messaggio", "")
        tag = data.get("tag", None)
        scope = data.get("scope", None)
    except Exception as e:
        print("Errore durante il parsing:", e)
        return {"error": "Error parsing JSON"}

    try:
        start_time = time.time()
        result = await asyncio.to_thread(
            conversation_chain.invoke,
            {"input": msg}
        )
        duration = time.time() - start_time
        print(f"[DEBUG] Tempo generazione risposta: {duration:.2f} sec")
        print(f"[DEBUG] Risultato ottenuto da LangChain: {result}")

        if isinstance(result, dict):
            response = result.get("output") or str(result)
        else:
            response = str(result)

    except Exception as e:
        print(f"[ERRORE GENERAZIONE AI] {repr(e)}")
        return {"error": f"Errore nella generazione della risposta AI: {str(e)}"}

    try:
        save_memory(msg, response, tag, scope)
    except Exception as e:
        print(f"[ERRORE SALVATAGGIO MEMORIA] {e}")
        return {"error": f"Errore nel salvataggio: {str(e)}"}

    return {
        "status": "ok",
        "tempo_risposta": f"{duration:.2f} secondi",
        "messaggio": response,
        "tag": tag,
        "scope": scope
    }

# =====================================
# Avvio del server
# =====================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

