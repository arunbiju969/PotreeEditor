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
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS edit_history ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,"
            " edit_type TEXT NOT NULL,"
            " potree_dir TEXT NOT NULL,"
            " las_path TEXT,"
            " target_itc INTEGER,"
            " assigned_itcs TEXT,"
            " id_mapping TEXT,"
            " segmented_las_path TEXT,"
            " points_affected INTEGER,"
            " details TEXT,"
            " status TEXT DEFAULT 'committed',"
            " reverted_at DATETIME"
            ")"
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

    # -------------------------------------------------------------------------
    # Edit History & Undo / Revert Logging
    # -------------------------------------------------------------------------
    def log_edit(self, edit_type: str, potree_dir: str, las_path: str = None,
                 target_itc: int = None, assigned_itcs: list = None, id_mapping: dict = None,
                 segmented_las_path: str = None, points_affected: int = 0,
                 details: dict = None) -> int:
        """Log an edit operation (octree patch, split, manual classification) into the database."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO edit_history ("
                " edit_type, potree_dir, las_path, target_itc, assigned_itcs, id_mapping,"
                " segmented_las_path, points_affected, details, status"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'committed')",
                (
                    str(edit_type),
                    str(potree_dir or ""),
                    str(las_path or "") if las_path else None,
                    int(target_itc) if target_itc is not None else None,
                    json.dumps(assigned_itcs or []),
                    json.dumps(id_mapping or {}),
                    str(segmented_las_path or "") if segmented_las_path else None,
                    int(points_affected or 0),
                    json.dumps(details or {}),
                )
            )
            self._conn.commit()
            return cur.lastrowid

    def get_edit_history(self, potree_dir: str = None, limit: int = 100) -> list:
        """Retrieve recent edit history from the database, newest first."""
        with self._lock:
            cur = self._conn.cursor()
            cols = [
                "id", "timestamp", "edit_type", "potree_dir", "las_path", "target_itc",
                "assigned_itcs", "id_mapping", "segmented_las_path", "points_affected",
                "details", "status", "reverted_at"
            ]
            col_str = ", ".join(cols)
            if potree_dir:
                cur.execute(
                    f"SELECT {col_str} FROM edit_history WHERE potree_dir = ? ORDER BY id DESC LIMIT ?",
                    (str(potree_dir), limit)
                )
            else:
                cur.execute(
                    f"SELECT {col_str} FROM edit_history ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
            rows = cur.fetchall()
            result = []
            for row in rows:
                item = dict(zip(cols, row))
                try:
                    item["assigned_itcs"] = json.loads(item["assigned_itcs"]) if item["assigned_itcs"] else []
                except Exception:
                    pass
                try:
                    item["id_mapping"] = json.loads(item["id_mapping"]) if item["id_mapping"] else {}
                except Exception:
                    pass
                try:
                    item["details"] = json.loads(item["details"]) if item["details"] else {}
                except Exception:
                    pass
                result.append(item)
            return result

    def mark_edit_reverted(self, edit_id: int) -> bool:
        """Mark an edit record as reverted in the database."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE edit_history SET status = 'reverted', reverted_at = CURRENT_TIMESTAMP WHERE id = ?",
                (int(edit_id),)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_edit_by_id(self, edit_id: int):
        """Fetch a single edit record by database ID."""
        with self._lock:
            cur = self._conn.cursor()
            cols = [
                "id", "timestamp", "edit_type", "potree_dir", "las_path", "target_itc",
                "assigned_itcs", "id_mapping", "segmented_las_path", "points_affected",
                "details", "status", "reverted_at"
            ]
            col_str = ", ".join(cols)
            cur.execute(f"SELECT {col_str} FROM edit_history WHERE id = ?", (int(edit_id),))
            row = cur.fetchone()
            if not row:
                return None
            item = dict(zip(cols, row))
            try:
                item["assigned_itcs"] = json.loads(item["assigned_itcs"]) if item["assigned_itcs"] else []
            except Exception:
                pass
            try:
                item["id_mapping"] = json.loads(item["id_mapping"]) if item["id_mapping"] else {}
            except Exception:
                pass
            try:
                item["details"] = json.loads(item["details"]) if item["details"] else {}
            except Exception:
                pass
            return item

    def close(self):
        with self._lock:
            self._conn.commit()
            self._conn.close()
