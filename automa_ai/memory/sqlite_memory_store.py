import json
import sqlite3
from datetime import datetime
from typing import Optional, List

from automa_ai.memory.memory_stores import BaseMemoryStore
from automa_ai.memory.memory_types import MemoryEntry, MemoryType


class SQLiteMemoryStore(BaseMemoryStore):
    """SQLite-based persistent memory storage."""

    @classmethod
    def from_config(cls, config: dict) -> "BaseMemoryStore":
        """
        store: {
            "db_path": str, Path to the database file
        }
        """
        db_path = config.get("db_path")
        assert db_path, "db_path must be defined for SQLiteMemoryStore."

        return cls(
            db_path = db_path
        )

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp REAL,
                    memory_type TEXT,
                    importance_score REAL,
                    access_count INTEGER,
                    last_accessed REAL
                )
            """)
            conn.commit()

    def write_memory(self, entries: List[MemoryEntry]) -> None:
        """Write a memory entry to SQLite storage."""
        with sqlite3.connect(self.db_path) as conn:
            data_to_insert = [
                (
                    entry.session_id,
                    entry.user_id,
                    entry.content,
                    json.dumps(entry.metadata),
                    entry.timestamp.timestamp(),
                    entry.memory_type.value,
                    entry.importance_score,
                    entry.access_count,
                    entry.last_accessed.timestamp()
                )
                for entry in entries
            ]

            conn.executemany("""
                        INSERT INTO memories 
                        (session_id, user_id, content, metadata, timestamp, memory_type, importance_score, access_count, last_accessed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, data_to_insert)
            conn.commit()

    def read_memories(
            self,
            query: Optional[str] = None,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
            memory_type: Optional[MemoryType] = None,
            limit: int = 10
    ) -> List[MemoryEntry]:
        """Read memory entries from SQLite storage."""
        sql = "SELECT * FROM memories WHERE "
        params = []

        if session_id:
            sql += "session_id = ? AND "
            params.append(session_id)

        if user_id:
            sql += "user_id = ? AND "
            params.append(user_id)

        if memory_type:
            sql += " memory_type = ?"
            params.append(memory_type.value)

        # Needs work - I don't think this will match anything
        #if query:
        #    sql += " AND (content LIKE ? OR metadata LIKE ?)"
        #    params.extend([f"%{query}%", f"%{query}%"])

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        memories = []
        for row in rows:
            print("retrieved row: ", row)
            entry = MemoryEntry(
                id=row[0],
                session_id=row[1],
                content=row[2],
                metadata=json.loads(row[3]) if row[3] else {},
                timestamp=datetime.fromtimestamp(row[4]),
                memory_type=MemoryType(row[5]),
                importance_score=row[6],
                access_count=row[7],
                last_accessed=datetime.fromtimestamp(row[8])
            )
            memories.append(entry)

        return memories

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_memories(self, memory_type: Optional[MemoryType] = None) -> None:
        """Clear memories of a specific type or all memories."""
        with sqlite3.connect(self.db_path) as conn:
            if memory_type is None:
                conn.execute("DELETE FROM memories")
            else:
                conn.execute("DELETE FROM memories WHERE memory_type = ?", (memory_type.value,))
            conn.commit()
