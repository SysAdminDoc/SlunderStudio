"""
Slunder Studio — Lyrics Database
SQLite storage for lyrics generation history, favorites, search, and version diffs.
"""
import sqlite3
import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, Optional, Sequence
from pathlib import Path
from dataclasses import dataclass, field

from core.settings import get_config_dir

# SQLite waits this long for a competing writer before raising "database is
# locked". WAL still serializes writers, so a busy timeout plus the in-process
# lock below is what keeps GUI and worker threads from failing each other.
BUSY_TIMEOUT_MS = 10_000
_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4)


@dataclass
class LyricsEntry:
    """A single lyrics generation record."""
    id: int = 0
    timestamp: float = 0.0
    prompt: str = ""
    genre: str = ""
    mood: str = ""
    language: str = "en"
    model_id: str = ""
    temperature: float = 0.8
    lyrics_original: str = ""
    lyrics_edited: str = ""
    structure_tags: str = ""
    is_favorite: bool = False
    rating: int = 0  # 0-5 stars
    notes: str = ""
    generation_params: str = "{}"  # JSON blob of all params

    @property
    def lyrics(self) -> str:
        """Return edited version if available, otherwise original."""
        return self.lyrics_edited if self.lyrics_edited else self.lyrics_original

    @property
    def has_edits(self) -> bool:
        return bool(self.lyrics_edited) and self.lyrics_edited != self.lyrics_original

    @property
    def timestamp_str(self) -> str:
        from datetime import datetime
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M")

    @property
    def preview(self) -> str:
        """First 100 chars of lyrics for list display."""
        text = self.lyrics.replace("\n", " ").strip()
        return text[:100] + "..." if len(text) > 100 else text


class LyricsDB:
    """SQLite database manager for lyrics history."""

    _instance: Optional["LyricsDB"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._db_path = get_config_dir() / "lyrics_history.db"
        self._conn: Optional[sqlite3.Connection] = None
        # Re-entrant so a public method may call another public method while
        # holding the lock (search() -> get_recent(), import_entries() -> save()).
        self._op_lock = threading.RLock()
        self._closed = False
        self._ensure_db()

    def _ensure_db(self):
        """Create database and tables if they don't exist."""
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS lyrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                prompt TEXT NOT NULL DEFAULT '',
                genre TEXT NOT NULL DEFAULT '',
                mood TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT 'en',
                model_id TEXT NOT NULL DEFAULT '',
                temperature REAL NOT NULL DEFAULT 0.8,
                lyrics_original TEXT NOT NULL DEFAULT '',
                lyrics_edited TEXT NOT NULL DEFAULT '',
                structure_tags TEXT NOT NULL DEFAULT '',
                is_favorite INTEGER NOT NULL DEFAULT 0,
                rating INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                generation_params TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_lyrics_timestamp ON lyrics(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_lyrics_genre ON lyrics(genre);
            CREATE INDEX IF NOT EXISTS idx_lyrics_favorite ON lyrics(is_favorite);
            CREATE INDEX IF NOT EXISTS idx_lyrics_rating ON lyrics(rating DESC);

            CREATE VIRTUAL TABLE IF NOT EXISTS lyrics_fts USING fts5(
                prompt, lyrics_original, lyrics_edited, genre, mood, notes,
                content='lyrics', content_rowid='id'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS lyrics_ai AFTER INSERT ON lyrics BEGIN
                INSERT INTO lyrics_fts(rowid, prompt, lyrics_original, lyrics_edited, genre, mood, notes)
                VALUES (new.id, new.prompt, new.lyrics_original, new.lyrics_edited, new.genre, new.mood, new.notes);
            END;

            CREATE TRIGGER IF NOT EXISTS lyrics_ad AFTER DELETE ON lyrics BEGIN
                INSERT INTO lyrics_fts(lyrics_fts, rowid, prompt, lyrics_original, lyrics_edited, genre, mood, notes)
                VALUES ('delete', old.id, old.prompt, old.lyrics_original, old.lyrics_edited, old.genre, old.mood, old.notes);
            END;

            CREATE TRIGGER IF NOT EXISTS lyrics_au AFTER UPDATE ON lyrics BEGIN
                INSERT INTO lyrics_fts(lyrics_fts, rowid, prompt, lyrics_original, lyrics_edited, genre, mood, notes)
                VALUES ('delete', old.id, old.prompt, old.lyrics_original, old.lyrics_edited, old.genre, old.mood, old.notes);
                INSERT INTO lyrics_fts(rowid, prompt, lyrics_original, lyrics_edited, genre, mood, notes)
                VALUES (new.id, new.prompt, new.lyrics_original, new.lyrics_edited, new.genre, new.mood, new.notes);
            END;
        """)
        self._conn.commit()

    # ── Connection and transaction plumbing ────────────────────────────────────

    @staticmethod
    def _is_locked(exc: sqlite3.OperationalError) -> bool:
        return "locked" in str(exc).lower() or "busy" in str(exc).lower()

    def _require_conn(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("Lyrics database is closed; call reopen() first")
        if self._conn is None:
            self._ensure_db()
        return self._conn

    def reopen(self):
        """Reconnect after close(). Safe to call on an already-open database."""
        with self._op_lock:
            if self._conn is not None:
                return
            self._closed = False
            self._ensure_db()

    @contextmanager
    def _write(self):
        """Serialized write transaction with a bounded retry on lock contention."""
        with self._op_lock:
            conn = self._require_conn()
            last_error: Optional[sqlite3.OperationalError] = None
            for delay in (*_RETRY_DELAYS, None):
                try:
                    conn.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as exc:
                    if delay is None or not self._is_locked(exc):
                        raise
                    last_error = exc
                    time.sleep(delay)
                    continue
                try:
                    yield conn
                except BaseException:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
                conn.execute("COMMIT")
                return
            raise last_error  # pragma: no cover - loop always returns or raises

    def _read(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Serialized read with a bounded retry on lock contention."""
        with self._op_lock:
            conn = self._require_conn()
            for delay in (*_RETRY_DELAYS, None):
                try:
                    return conn.execute(sql, params).fetchall()
                except sqlite3.OperationalError as exc:
                    if delay is None or not self._is_locked(exc):
                        raise
                    time.sleep(delay)
        return []

    def _row_to_entry(self, row: sqlite3.Row) -> LyricsEntry:
        """Convert a database row to a LyricsEntry."""
        return LyricsEntry(
            id=row["id"],
            timestamp=row["timestamp"],
            prompt=row["prompt"],
            genre=row["genre"],
            mood=row["mood"],
            language=row["language"],
            model_id=row["model_id"],
            temperature=row["temperature"],
            lyrics_original=row["lyrics_original"],
            lyrics_edited=row["lyrics_edited"],
            structure_tags=row["structure_tags"],
            is_favorite=bool(row["is_favorite"]),
            rating=row["rating"],
            notes=row["notes"],
            generation_params=row["generation_params"],
        )

    # ── CRUD ───────────────────────────────────────────────────────────────────

    _INSERT_SQL = """
            INSERT INTO lyrics (
                timestamp, prompt, genre, mood, language, model_id, temperature,
                lyrics_original, lyrics_edited, structure_tags, is_favorite,
                rating, notes, generation_params
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    @staticmethod
    def _insert_params(entry: LyricsEntry) -> tuple:
        return (
            entry.timestamp, entry.prompt, entry.genre, entry.mood,
            entry.language, entry.model_id, entry.temperature,
            entry.lyrics_original, entry.lyrics_edited, entry.structure_tags,
            int(entry.is_favorite), entry.rating, entry.notes,
            entry.generation_params,
        )

    def save(self, entry: LyricsEntry) -> int:
        """Save a new lyrics entry. Returns the entry ID."""
        if entry.timestamp == 0:
            entry.timestamp = time.time()

        with self._write() as conn:
            cursor = conn.execute(self._INSERT_SQL, self._insert_params(entry))
            entry.id = cursor.lastrowid
        return entry.id

    def save_many(self, entries: Iterable[LyricsEntry]) -> list[int]:
        """Import several entries in one transaction. All or nothing."""
        batch = list(entries)
        if not batch:
            return []
        now = time.time()
        for entry in batch:
            if entry.timestamp == 0:
                entry.timestamp = now
        with self._write() as conn:
            for entry in batch:
                cursor = conn.execute(self._INSERT_SQL, self._insert_params(entry))
                entry.id = cursor.lastrowid
        return [entry.id for entry in batch]

    def update(self, entry: LyricsEntry):
        """Update an existing entry."""
        with self._write() as conn:
            conn.execute("""
                UPDATE lyrics SET
                    prompt=?, genre=?, mood=?, language=?, model_id=?, temperature=?,
                    lyrics_original=?, lyrics_edited=?, structure_tags=?, is_favorite=?,
                    rating=?, notes=?, generation_params=?
                WHERE id=?
            """, (
                entry.prompt, entry.genre, entry.mood, entry.language,
                entry.model_id, entry.temperature, entry.lyrics_original,
                entry.lyrics_edited, entry.structure_tags, int(entry.is_favorite),
                entry.rating, entry.notes, entry.generation_params, entry.id,
            ))

    def delete(self, entry_id: int):
        """Delete an entry by ID."""
        with self._write() as conn:
            conn.execute("DELETE FROM lyrics WHERE id=?", (entry_id,))

    def get(self, entry_id: int) -> Optional[LyricsEntry]:
        """Get a single entry by ID."""
        rows = self._read("SELECT * FROM lyrics WHERE id=?", (entry_id,))
        return self._row_to_entry(rows[0]) if rows else None

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 50, offset: int = 0) -> list[LyricsEntry]:
        """Get recent entries, newest first."""
        rows = self._read(
            "SELECT * FROM lyrics ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [self._row_to_entry(r) for r in rows]

    def get_favorites(self, limit: int = 50) -> list[LyricsEntry]:
        """Get favorited entries."""
        rows = self._read(
            "SELECT * FROM lyrics WHERE is_favorite=1 ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_entry(r) for r in rows]

    def get_by_genre(self, genre: str, limit: int = 50) -> list[LyricsEntry]:
        """Get entries by genre."""
        rows = self._read(
            "SELECT * FROM lyrics WHERE genre=? ORDER BY timestamp DESC LIMIT ?",
            (genre, limit),
        )
        return [self._row_to_entry(r) for r in rows]

    def search(self, query: str, limit: int = 50) -> list[LyricsEntry]:
        """Full-text search across prompts, lyrics, genre, mood, notes."""
        if not query.strip():
            return self.get_recent(limit)
        try:
            rows = self._read("""
                SELECT l.* FROM lyrics l
                JOIN lyrics_fts f ON l.id = f.rowid
                WHERE lyrics_fts MATCH ?
                ORDER BY l.timestamp DESC LIMIT ?
            """, (query, limit))
            return [self._row_to_entry(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS query syntax error — fall back to LIKE
            pattern = f"%{query}%"
            rows = self._read("""
                SELECT * FROM lyrics
                WHERE prompt LIKE ? OR lyrics_original LIKE ? OR lyrics_edited LIKE ?
                    OR genre LIKE ? OR mood LIKE ? OR notes LIKE ?
                ORDER BY timestamp DESC LIMIT ?
            """, (pattern, pattern, pattern, pattern, pattern, pattern, limit))
            return [self._row_to_entry(r) for r in rows]

    def toggle_favorite(self, entry_id: int) -> bool:
        """Toggle favorite status. Returns new state."""
        with self._write() as conn:
            row = conn.execute(
                "SELECT is_favorite FROM lyrics WHERE id=?", (entry_id,)
            ).fetchone()
            if row is None:
                return False
            new_val = 0 if row["is_favorite"] else 1
            conn.execute("UPDATE lyrics SET is_favorite=? WHERE id=?", (new_val, entry_id))
        return bool(new_val)

    def set_rating(self, entry_id: int, rating: int):
        """Set star rating (0-5)."""
        rating = max(0, min(5, rating))
        with self._write() as conn:
            conn.execute("UPDATE lyrics SET rating=? WHERE id=?", (rating, entry_id))

    def get_genres(self) -> list[str]:
        """Get all unique genres in the database."""
        rows = self._read(
            "SELECT DISTINCT genre FROM lyrics WHERE genre != '' ORDER BY genre"
        )
        return [r["genre"] for r in rows]

    def count(self) -> int:
        """Total number of entries."""
        rows = self._read("SELECT COUNT(*) as c FROM lyrics")
        return rows[0]["c"] if rows else 0

    def close(self):
        """Close the database connection once in-flight operations finish."""
        with self._op_lock:
            self._closed = True
            if self._conn:
                self._conn.close()
                self._conn = None
