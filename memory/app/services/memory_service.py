import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MemoryRecord:
    id: str
    memory: str
    user_id: str
    agent_id: Optional[str]
    run_id: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: str
    updated_at: Optional[str]


class MemoryService:
    """SQLite-backed storage engine compatible with Mem0 API expectations."""

    def __init__(self, db_path: str = "memory.db") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        expected_columns = {
            "id",
            "memory",
            "user_id",
            "agent_id",
            "run_id",
            "metadata",
            "created_at",
            "updated_at",
        }
        with self._lock:
            info = self._conn.execute("PRAGMA table_info(memories)").fetchall()
            if info:
                column_names = {row[1] for row in info}
                if column_names != expected_columns:
                    self._conn.execute("DROP TABLE IF EXISTS memories")
                    self._conn.commit()
                    info = []
            if not info:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        memory TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        agent_id TEXT,
                        run_id TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT
                    )
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)"
                )
                self._conn.commit()

    def add_memories(
        self,
        messages: Iterable[Dict[str, Any]],
        user_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        metadata_json = json.dumps(metadata) if metadata is not None else None
        now = self._now()
        results: List[Dict[str, str]] = []
        records = [self._message_to_memory_fragment(message) for message in messages]
        with self._lock:
            for memory_text in records:
                memory_id = str(uuid.uuid4())
                self._conn.execute(
                    """
                    INSERT INTO memories (id, memory, user_id, agent_id, run_id, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        memory_id,
                        memory_text,
                        user_id,
                        agent_id,
                        run_id,
                        metadata_json,
                        now,
                    ),
                )
                results.append({"id": memory_id, "memory": memory_text, "event": "ADD"})
            self._conn.commit()
        return results

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_memories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        records = [self._row_to_record(row) for row in self._fetch_all_rows()]
        filtered = [
            record for record in records if self._matches_filters(record, filters)
        ]
        total = len(filtered)
        start = max(page - 1, 0) * page_size
        end = start + page_size
        page_items = filtered[start:end]
        return {
            "count": total,
            "next": None if end >= total else "",
            "previous": None if start <= 0 else "",
            "results": [self._record_to_dict(record) for record in page_items],
        }

    def search_memories(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        records = [self._row_to_record(row) for row in self._fetch_all_rows()]
        filtered = [
            record for record in records if self._matches_filters(record, filters)
        ]
        scored: List[Dict[str, Any]] = []
        for record in filtered:
            score = self._score_memory(record.memory, query)
            if score <= 0:
                continue
            payload = self._record_to_dict(record)
            payload["score"] = score
            scored.append(payload)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryRecord]:
        current = self.get_memory(memory_id)
        if not current:
            return None
        new_text = text if text is not None else current.memory
        new_metadata = metadata if metadata is not None else current.metadata
        metadata_json = json.dumps(new_metadata) if new_metadata is not None else None
        updated_at = self._now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE memories
                SET memory = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_text, metadata_json, updated_at, memory_id),
            )
            self._conn.commit()
        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def _fetch_all_rows(self) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM memories").fetchall()

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
        return MemoryRecord(
            id=row["id"],
            memory=row["memory"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def serialize(self, record: MemoryRecord) -> Dict[str, Any]:
        return {
            "id": record.id,
            "memory": record.memory,
            "user_id": record.user_id,
            "agent_id": record.agent_id,
            "run_id": record.run_id,
            "metadata": record.metadata,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def _record_to_dict(self, record: MemoryRecord) -> Dict[str, Any]:
        return self.serialize(record)

    def _matches_filters(
        self, record: MemoryRecord, filters: Optional[Dict[str, Any]]
    ) -> bool:
        if not filters:
            return True
        if "AND" in filters:
            return all(self._matches_filters(record, item) for item in filters["AND"])
        if "OR" in filters:
            return any(self._matches_filters(record, item) for item in filters["OR"])
        if "NOT" in filters:
            return not any(
                self._matches_filters(record, item) for item in filters["NOT"]
            )

        for key, condition in filters.items():
            if not self._match_condition(record, key, condition):
                return False
        return True

    def _match_condition(self, record: MemoryRecord, key: str, condition: Any) -> bool:
        value = self._extract_field(record, key)
        if condition == "*":
            return True
        if not isinstance(condition, dict):
            condition = {"eq": condition}
        for operator, expected in condition.items():
            if operator == "eq" and value != expected:
                return False
            elif operator == "ne" and value == expected:
                return False
            elif operator == "gt" and not (value is not None and value > expected):
                return False
            elif operator == "gte" and not (value is not None and value >= expected):
                return False
            elif operator == "lt" and not (value is not None and value < expected):
                return False
            elif operator == "lte" and not (value is not None and value <= expected):
                return False
            elif operator == "in":
                if isinstance(expected, (list, tuple, set)):
                    if value not in expected:
                        return False
                else:
                    return False
            elif operator == "icontains":
                if not (isinstance(value, str) and isinstance(expected, str)):
                    return False
                if expected.lower() not in value.lower():
                    return False
            elif operator not in {
                "eq",
                "ne",
                "gt",
                "gte",
                "lt",
                "lte",
                "in",
                "icontains",
            }:
                return False
        return True

    def _extract_field(self, record: MemoryRecord, dotted_key: str) -> Any:
        data: Any = self._record_to_dict(record)
        for part in dotted_key.split("."):
            if isinstance(data, dict):
                data = data.get(part)
            else:
                data = None
            if data is None:
                break
        return data

    def _message_to_memory_fragment(self, message: Dict[str, Any]) -> str:
        if hasattr(message, "dict"):
            payload = message.dict()
        elif isinstance(message, dict):
            payload = message
        else:
            payload = {
                "role": getattr(message, "role", "user"),
                "content": getattr(message, "content", ""),
            }
        role = payload.get("role", "user")
        content = payload.get("content", "")
        return f"{role}: {content}".strip()

    def _score_memory(self, memory_text: str, query: str) -> float:
        if not memory_text or not query:
            return 0.0
        return SequenceMatcher(None, query.lower(), memory_text.lower()).ratio()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
