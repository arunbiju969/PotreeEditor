"""Tiny SQLite-backed key/value settings store for persisted UI settings."""
import json
import os
import sqlite3
import threading


class SettingsStore:
    """Persistent key/value store (JSON-encoded values) in a SQLite file.

    The app writes a handful of settings on debounced change and reads them
    once at startup. A single lock makes it safe to touch from any thread.
    """

    def __init__(self, path: str):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS settings ("
            " key TEXT PRIMARY KEY,"
            " value TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str, default=None):
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (ValueError, TypeError):
            return default

    def set(self, key: str, value):
        self.set_many({key: value})

    def set_many(self, mapping: dict):
        """Upsert several settings in one transaction."""
        with self._lock:
            self._conn.executemany(
                "INSERT INTO settings(key, value) VALUES(?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(k, json.dumps(v)) for k, v in mapping.items()],
            )
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.commit()
            self._conn.close()
