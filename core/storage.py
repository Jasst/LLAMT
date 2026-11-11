import sqlite3, uuid, time
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sessions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    system_prompt TEXT,
    created_at REAL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    parent_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL,
    selected_child_id TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent  ON nodes(parent_id);
"""

# единое соединение (Gradio дёргает из разных потоков → check_same_thread=False)
_CONN: Optional[sqlite3.Connection] = None

def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _CONN.execute("PRAGMA foreign_keys = ON")
        _CONN.execute("PRAGMA journal_mode = WAL")
    return _CONN

def init_db():
    c = _conn()
    c.executescript(SCHEMA)
    c.commit()

# ─── сессии ───
def create_session(title="Новый чат", system_prompt="") -> str:
    sid = uuid.uuid4().hex
    c = _conn()
    c.execute("INSERT INTO sessions VALUES (?,?,?,?,?)",
              (sid, title, system_prompt, time.time(), time.time()))
    c.commit()
    return sid

def list_sessions() -> List[Dict]:
    rows = _conn().execute(
        "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]

def rename_session(sid, title):
    c = _conn()
    c.execute("UPDATE sessions SET title=?, updated_at=? WHERE id=?",
              (title, time.time(), sid))
    c.commit()

def delete_session(sid):
    c = _conn()
    c.execute("DELETE FROM nodes WHERE session_id=?", (sid,))
    c.execute("DELETE FROM sessions WHERE id=?", (sid,))
    c.commit()

def get_session(sid) -> Optional[Dict]:
    r = _conn().execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    return dict(r) if r else None

# ─── узлы ───
def add_node(session_id, parent_id, role, content) -> str:
    nid = uuid.uuid4().hex
    c = _conn()
    c.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,NULL)",
              (nid, session_id, parent_id, role, content, time.time()))
    if parent_id:
        c.execute("UPDATE nodes SET selected_child_id=? WHERE id=?",
                  (nid, parent_id))
    c.execute("UPDATE sessions SET updated_at=? WHERE id=?",
              (time.time(), session_id))
    c.commit()
    return nid

def update_node_content(nid, content):
    c = _conn()
    c.execute("UPDATE nodes SET content=? WHERE id=?", (content, nid))
    c.commit()

def set_selected_child(parent_id, child_id):
    c = _conn()
    c.execute("UPDATE nodes SET selected_child_id=? WHERE id=?",
              (child_id, parent_id))
    c.commit()

def get_node(nid) -> Optional[Dict]:
    r = _conn().execute("SELECT * FROM nodes WHERE id=?", (nid,)).fetchone()
    return dict(r) if r else None

def get_active_path(session_id) -> List[Dict]:
    c = _conn()
    root = c.execute(
        "SELECT * FROM nodes WHERE session_id=? AND parent_id IS NULL",
        (session_id,)).fetchone()
    if not root:
        return []
    path = [dict(root)]
    cur = root
    while cur["selected_child_id"]:
        nxt = c.execute("SELECT * FROM nodes WHERE id=?",
                        (cur["selected_child_id"],)).fetchone()
        if not nxt:
            break
        path.append(dict(nxt))
        cur = nxt
    return path

def get_siblings(node_id) -> List[Dict]:
    n = get_node(node_id)
    if not n:
        return []
    c = _conn()
    if n["parent_id"] is None:
        rows = c.execute(
            "SELECT * FROM nodes WHERE session_id=? AND parent_id IS NULL "
            "ORDER BY created_at", (n["session_id"],)).fetchall()
    else:
        rows = c.execute(
            "SELECT * FROM nodes WHERE parent_id=? ORDER BY created_at",
            (n["parent_id"],)).fetchall()
    return [dict(r) for r in rows]