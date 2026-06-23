import sqlite3
import uuid
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def _data_dir() -> Path:
    d = Path(os.environ.get("APP_DATA_DIR", str(Path(__file__).parent.parent / "data")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db_path() -> str:
    return str(_data_dir() / "notebooks.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS notebooks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_opened_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                citations TEXT DEFAULT '[]',
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            );
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Notebooks ────────────────────────────────────────────────────────────────

def create_notebook(name: str) -> dict:
    nb_id = str(uuid.uuid4())
    ts = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO notebooks (id, name, created_at, last_opened_at) VALUES (?, ?, ?, ?)",
            (nb_id, name, ts, ts),
        )
    return {"id": nb_id, "name": name, "created_at": ts, "last_opened_at": ts}


def list_notebooks() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM notebooks ORDER BY last_opened_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_notebook(nb_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM notebooks WHERE id = ?", (nb_id,)).fetchone()
    return dict(row) if row else None


def update_notebook_name(nb_id: str, name: str):
    with get_conn() as conn:
        conn.execute("UPDATE notebooks SET name = ? WHERE id = ?", (name, nb_id))


def touch_notebook(nb_id: str):
    with get_conn() as conn:
        conn.execute("UPDATE notebooks SET last_opened_at = ? WHERE id = ?", (_now(), nb_id))


def delete_notebook(nb_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM notebooks WHERE id = ?", (nb_id,))


# ── Chat ─────────────────────────────────────────────────────────────────────

def get_chat_history(nb_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE notebook_id = ? ORDER BY id ASC",
            (nb_id,),
        ).fetchall()
    result = []
    for row in rows:
        r = dict(row)
        r["citations"] = json.loads(r.get("citations") or "[]")
        result.append(r)
    return result


def add_chat_message(nb_id: str, role: str, message: str, citations: list | None = None) -> dict:
    ts = _now()
    citations_json = json.dumps(citations or [])
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO chat_messages (notebook_id, role, message, timestamp, citations) "
            "VALUES (?, ?, ?, ?, ?)",
            (nb_id, role, message, ts, citations_json),
        )
        msg_id = cursor.lastrowid
    return {
        "id": msg_id,
        "notebook_id": nb_id,
        "role": role,
        "message": message,
        "timestamp": ts,
        "citations": citations or [],
    }


def clear_chat_history(nb_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_messages WHERE notebook_id = ?", (nb_id,))
