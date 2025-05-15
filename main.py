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
import html
import asyncio

from rag_utils import chunk_text, build_rag_prompt, call_model_and_get_diff
 
from fastapi import HTTPException

from sentence_transformers import SentenceTransformer
import json
from rag_utils import chunk_text, build_rag_prompt

# LangChain + Ollama definitivo
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from file_versioning import router as versioning_router
from file_versioning import save_file_version
from file_management import router as file_router
from memory_embeddings import embed_all_records
from memory_embeddings import search_similar_memories
from starlette.websockets import WebSocketState

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
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        chat_id TEXT PRIMARY KEY,
        nome TEXT
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
        # Escape HTML per evitare XSS, mantenendo intatti i newline
        prompt   = html.escape(prompt)
        response = html.escape(response)

        # Calcola l’embedding (se hai già il modello caricato altrove, usa quello)
        embedding_model_local = SentenceTransformer("all-MiniLM-L6-v2")
        combined_text = f"{prompt}\n\n{response}"
        embedding = embedding_model_local.encode(combined_text)
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

        # Inserimento in SQLite
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO memories
              (prompt, response, tag, scope, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
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

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # oppure ["http://localhost:5500"] se usi live server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        simili = [r for r in search_similar_memories(msg, top_k=1) if r["similarity"] >= 0.45][:3]
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
    
@app.get("/storico/{chat_id}")
async def get_chat_history(chat_id: str):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("SELECT prompt, response FROM memories WHERE scope = ? ORDER BY id ASC", (chat_id,))
        rows = cur.fetchall()
        conn.close()
        return [{"prompt": row[0], "response": row[1]} for row in rows]
    except Exception as e:
        print(f"[ERRORE STORICO CHAT] {e}")
        return {"error": str(e)}
   

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        current_chat = "default"
        while True:
            message = await websocket.receive_text()
            if message.startswith("switch_chat:"):
                current_chat = message.split(":", 1)[1]
                continue

            raw = message

            # 1) Chunking
            chunks = chunk_text(raw)  # default max_tokens=1500

            full_code = ""
            # 2) Per ogni chunk, genera il codice ottimizzato in streaming
            for idx, chunk in enumerate(chunks, start=1):
                # istruzioni al modello: solo il codice ottimizzato di questo chunk
                prompt_i = (
                    f"[CHUNK {idx}/{len(chunks)}]\n"
                    "Ecco il seguente chunk di codice da ottimizzare:\n"
                    f"{chunk}\n"
                    "Restituisci *solo* il codice ottimizzato, senza spiegazioni."
                )
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "localmistralinstruct",
                            "prompt": prompt_i,
                            "stream": True
                        }
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line.split("data:")[-1])
                                tok = data.get("response", "")
                                if tok:
                                    full_code += tok
                                    # streamma subito al frontend
                                    await websocket.send_text(json.dumps({"response": tok}))
                            except:
                                pass

            # 3) Marco fine chunking
            await websocket.send_text(json.dumps({"response": "[FINE]"}))

            # 4) Invia il file completo in un unico blocco
            await websocket.send_text(json.dumps({"response": full_code}))
            
            # 5) Marco fine completo
            await websocket.send_text(json.dumps({"response": "[FINE COMPLETO]"}))

            # 6) Salva in memoria raw_prompt e full_code
            save_memory(raw, full_code, scope=current_chat)

    except Exception as e:
        print("[WebSocket] Connessione chiusa:", e)
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



# =====================================
# API REST per gestione Chat
# =====================================
class Chat(BaseModel):
    chat_id: str
    nome: str

@app.get("/chats")
def get_chats():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT chat_id, nome FROM chats")
        chats = [{"chat_id": row[0], "nome": row[1]} for row in cursor.fetchall()]
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/chats")
def create_chat(chat: Chat):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chats (chat_id, nome) VALUES (?, ?)", (chat.chat_id, chat.nome))
        conn.commit()
        return {"status": "ok", "chat_id": chat.chat_id}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Chat con lo stesso ID già esistente.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.put("/chats/{chat_id}")
def rename_chat(chat_id: str, new_name: str = Body(..., embed=True)):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE chats SET nome = ? WHERE chat_id = ?", (new_name, chat_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat non trovata.")
        conn.commit()
        return {"status": "renamed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat non trovata.")
        conn.commit()
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
