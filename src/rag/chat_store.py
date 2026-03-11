# rag/chat_store.py
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)

class ChatStore:
    """Tiny SQLite wrapper to store chats and messages (good for demos)."""
    def __init__(self, db_path: Path | str = Path("data/chat_history.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._bootstrap()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys=ON;")
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _bootstrap(self) -> None:
        con = self._connect()
        cur = con.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS chats(
          id TEXT PRIMARY KEY,
          title TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          chat_id TEXT NOT NULL,
          role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
          content TEXT NOT NULL,
          tokens INTEGER,
          created_at TEXT NOT NULL,
          FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS feedback(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          chat_id TEXT,
          question TEXT NOT NULL,
          answer TEXT NOT NULL,
          rating INTEGER NOT NULL CHECK(rating IN (1, -1)),
          created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_chat_time ON messages(chat_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_chat_role ON messages(chat_id, role);
        CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
        """)
        con.commit()
        con.close()

    def create_chat(self, chat_id: str, title: str = "Untitled") -> str:
        con = self._connect()
        con.execute(
            "INSERT OR IGNORE INTO chats(id, title, created_at) VALUES (?, ?, ?);",
            (chat_id, title, utc_now())
        )
        con.commit()
        con.close()
        return chat_id

    def add_message(self, chat_id: str, role: str, content: str, tokens: Optional[int]=None) -> int:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO messages(chat_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?);",
            (chat_id, role, content, tokens, utc_now())
        )
        mid = cur.lastrowid
        con.commit()
        con.close()
        return int(mid)

    def get_messages(self, chat_id: str, limit: Optional[int]=None) -> List[Dict]:
        con = self._connect()
        cur = con.cursor()
        q = "SELECT id, role, content, tokens, created_at FROM messages WHERE chat_id=? ORDER BY created_at ASC"
        if limit:
            q += " LIMIT ?"
            cur.execute(q, (chat_id, limit))
        else:
            cur.execute(q, (chat_id,))
        rows = cur.fetchall()
        con.close()
        return [{"id": r[0], "role": r[1], "content": r[2], "tokens": r[3], "created_at": r[4]} for r in rows]

    def store_feedback(self, question: str, answer: str, rating: int, chat_id: Optional[str] = None) -> int:
        """Store user feedback for a Q&A pair. rating: 1 = helpful, -1 = not helpful."""
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO feedback(chat_id, question, answer, rating, created_at) VALUES (?, ?, ?, ?, ?);",
            (chat_id, question, answer, rating, utc_now())
        )
        fid = cur.lastrowid
        con.commit()
        con.close()
        return int(fid)

    def get_feedback(self, limit: Optional[int] = None) -> List[Dict]:
        """Retrieve stored feedback entries, most recent first."""
        con = self._connect()
        cur = con.cursor()
        q = "SELECT id, chat_id, question, answer, rating, created_at FROM feedback ORDER BY created_at DESC"
        if limit:
            q += " LIMIT ?"
            cur.execute(q, (limit,))
        else:
            cur.execute(q)
        rows = cur.fetchall()
        con.close()
        return [
            {"id": r[0], "chat_id": r[1], "question": r[2], "answer": r[3], "rating": r[4], "created_at": r[5]}
            for r in rows
        ]