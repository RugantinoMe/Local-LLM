from fastapi import APIRouter, HTTPException, Query
import sqlite3
import difflib

router = APIRouter()

DB_PATH = "G:/PROGETTI/Local LLM/LobeChat.db"

# =====================================
# Creazione della tabella file_versions
# =====================================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS file_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        content TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
''')
conn.commit()
conn.close()

# =====================================
# Funzione per salvare una versione
# =====================================
def save_file_version(filename: str, content: str):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO file_versions (filename, content) VALUES (?, ?)",
        (filename, content)
    )
    conn.commit()
    conn.close()

# =====================================
# Endpoint per elencare tutti i file tracciati
# =====================================
@router.get("/files")
def list_tracked_files():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT filename FROM file_versions")
        files = [row[0] for row in cursor.fetchall()]
        conn.close()
        return {"tracked_files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================
# Endpoint per confrontare due versioni
# =====================================
@router.get("/versioning/diff")
def diff_versions(filename: str, version1: int = Query(...), version2: int = Query(...)):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content FROM file_versions
            WHERE filename = ? AND id IN (?, ?)
            ORDER BY id
        """, (filename, version1, version2))
        rows = cursor.fetchall()
        conn.close()

        if len(rows) != 2:
            raise HTTPException(status_code=404, detail="Una o entrambe le versioni non trovate")

        content1, content2 = rows[0][0], rows[1][0]
        diff = list(difflib.unified_diff(
            content1.splitlines(),
            content2.splitlines(),
            lineterm="",
            fromfile=f"v{version1}",
            tofile=f"v{version2}"
        ))
        return {"diff": diff}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Fine del file file_versioning.py

# Ultima riga:
print("[DEBUG] file_versioning.py caricato correttamente.")
  

