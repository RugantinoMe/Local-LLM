import sqlite3
from fastapi import APIRouter, Request
from pydantic import BaseModel
import time
import os

# Configurazione database
DB_PATH = "G:/PROGETTI/Local LLM/LobeChat.db"

router = APIRouter()

# Modelli dati
class SaveFileRequest(BaseModel):
    project_name: str
    file_name: str
    file_path: str
    content: str

class SaveErrorRequest(BaseModel):
    project_name: str
    file_name: str
    error_message: str
    context: str = None

# Funzione per ottenere ID progetto esistente o crearlo
def get_or_create_project_id(project_name):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM progetti WHERE nome = ?", (project_name,))
    project = cursor.fetchone()
    if project:
        project_id = project[0]
    else:
        cursor.execute("INSERT INTO progetti (nome) VALUES (?)", (project_name,))
        conn.commit()
        project_id = cursor.lastrowid

    conn.close()
    return project_id

# Endpoint per salvare un file
@router.post("/save_file")
async def save_file(data: SaveFileRequest):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    try:
        project_id = get_or_create_project_id(data.project_name)

        # Salva il file nella tabella file
        cursor.execute(
            """
            INSERT INTO file (progetto_id, nome_file, percorso)
            VALUES (?, ?, ?)
            """,
            (project_id, data.file_name, data.file_path)
        )
        conn.commit()
        file_id = cursor.lastrowid

        # Salva il contenuto nella tabella file_versions
        cursor.execute(
            """
            INSERT INTO file_versions (filename, version, content)
            VALUES (?, ?, ?)
            """,
            (data.file_name, 1, data.content)
        )
        conn.commit()

        return {"status": "ok", "file_id": file_id}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

# Endpoint per salvare un errore
@router.post("/save_error")
async def save_error(data: SaveErrorRequest):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    try:
        project_id = get_or_create_project_id(data.project_name)

        # Cerca il file_id associato
        cursor.execute("SELECT id FROM file WHERE nome_file = ?", (data.file_name,))
        file = cursor.fetchone()
        file_id = file[0] if file else None

        # Inserisci l'errore
        cursor.execute(
            """
            INSERT INTO error_logs (errore, contesto, progetto_id, file_id)
            VALUES (?, ?, ?, ?)
            """,
            (data.error_message, data.context, project_id, file_id)
        )
        conn.commit()

        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()
