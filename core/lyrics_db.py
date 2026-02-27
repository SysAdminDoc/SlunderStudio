"""
Slunder Studio v0.0.2 — Lyrics Database
SQLite storage for lyrics generation history, favorites, search, and version diffs.
"""
import sqlite3
import json
import time
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

from core.settings import get_config_dir


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

    def __new__(cls):
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
        self._ensure_db()

    def _ensure_db(self):
        """Create database and tables if they don't exist."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

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

    def save(self, entry: LyricsEntry) -> int:
        """Save a new lyrics entry. Returns the entry ID."""
        if entry.timestamp == 0:
            entry.timestamp = time.time()

        cursor = self._conn.execute("""
            INSERT INTO lyrics (
                timestamp, prompt, genre, mood, language, model_id, temperature,
                lyrics_original, lyrics_edited, structure_tags, is_favorite,
                rating, notes, generation_params
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.timestamp, entry.prompt, entry.genre, entry.mood,
            entry.language, entry.model_id, entry.temperature,
            entry.lyrics_original, entry.lyrics_edited, entry.structure_tags,
            int(entry.is_favorite), entry.rating, entry.notes,
            entry.generation_params,
        ))
        self._conn.commit()
        entry.id = cursor.lastrowid
        return entry.id

    def update(self, entry: LyricsEntry):
        """Update an existing entry."""
        self._conn.execute("""
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
        self._conn.commit()

    def delete(self, entry_id: int):
        """Delete an entry by ID."""
        self._conn.execute("DELETE FROM lyrics WHERE id=?", (entry_id,))
        self._conn.commit()

    def get(self, entry_id: int) -> Optional[LyricsEntry]:
        """Get a single entry by ID."""
        row = self._conn.execute("SELECT * FROM lyrics WHERE id=?", (entry_id,)).fetchone()
        return self._row_to_entry(row) if row else None

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 50, offset: int = 0) -> list[LyricsEntry]:
        """Get recent entries, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM lyrics ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_favorites(self, limit: int = 50) -> list[LyricsEntry]:
        """Get favorited entries."""
        rows = self._conn.execute(
            "SELECT * FROM lyrics WHERE is_favorite=1 ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_by_genre(self, genre: str, limit: int = 50) -> list[LyricsEntry]:
        """Get entries by genre."""
        rows = self._conn.execute(
            "SELECT * FROM lyrics WHERE genre=? ORDER BY timestamp DESC LIMIT ?",
            (genre, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def search(self, query: str, limit: int = 50) -> list[LyricsEntry]:
        """Full-text search across prompts, lyrics, genre, mood, notes."""
        if not query.strip():
            return self.get_recent(limit)
        try:
            rows = self._conn.execute("""
                SELECT l.* FROM lyrics l
                JOIN lyrics_fts f ON l.id = f.rowid
                WHERE lyrics_fts MATCH ?
                ORDER BY l.timestamp DESC LIMIT ?
            """, (query, limit)).fetchall()
            return [self._row_to_entry(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS query syntax error — fall back to LIKE
            pattern = f"%{query}%"
            rows = self._conn.execute("""
                SELECT * FROM lyrics
                WHERE prompt LIKE ? OR lyrics_original LIKE ? OR lyrics_edited LIKE ?
                    OR genre LIKE ? OR mood LIKE ? OR notes LIKE ?
                ORDER BY timestamp DESC LIMIT ?
            """, (pattern, pattern, pattern, pattern, pattern, pattern, limit)).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def toggle_favorite(self, entry_id: int) -> bool:
        """Toggle favorite status. Returns new state."""
        row = self._conn.execute("SELECT is_favorite FROM lyrics WHERE id=?", (entry_id,)).fetchone()
        if row is None:
            return False
        new_val = 0 if row["is_favorite"] else 1
        self._conn.execute("UPDATE lyrics SET is_favorite=? WHERE id=?", (new_val, entry_id))
        self._conn.commit()
        return bool(new_val)

    def set_rating(self, entry_id: int, rating: int):
        """Set star rating (0-5)."""
        rating = max(0, min(5, rating))
        self._conn.execute("UPDATE lyrics SET rating=? WHERE id=?", (rating, entry_id))
        self._conn.commit()

    def get_genres(self) -> list[str]:
        """Get all unique genres in the database."""
        rows = self._conn.execute(
            "SELECT DISTINCT genre FROM lyrics WHERE genre != '' ORDER BY genre"
        ).fetchall()
        return [r["genre"] for r in rows]

    def count(self) -> int:
        """Total number of entries."""
        row = self._conn.execute("SELECT COUNT(*) as c FROM lyrics").fetchone()
        return row["c"] if row else 0

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
