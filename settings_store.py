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
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tree_edits_index ("
            " itc_id INTEGER PRIMARY KEY,"
            " parent_itc INTEGER NOT NULL,"
            " cluster_idx INTEGER NOT NULL,"
            " edit_id INTEGER NOT NULL,"
            " las_path TEXT NOT NULL,"
            " point_count INTEGER NOT NULL,"
            " status TEXT DEFAULT 'active'"
            ")"
        )
        # Snapshot of the staged (uncommitted) edit registry, so the sidebar
        # table is restored when the master point cloud is reopened.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS staged_edits ("
            " idx INTEGER PRIMARY KEY AUTOINCREMENT,"
            " payload TEXT NOT NULL)"
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

    # -------------------------------------------------------------------------
    # Tree Edits Overlay Index
    # -------------------------------------------------------------------------
    def register_tree_edit(self, itc_id: int, parent_itc: int, cluster_idx: int,
                           edit_id: int, las_path: str, point_count: int = 0):
        """Register an edited / split tree in the fast overlay index."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO tree_edits_index ("
                " itc_id, parent_itc, cluster_idx, edit_id, las_path, point_count, status"
                ") VALUES (?, ?, ?, ?, ?, ?, 'active')"
                " ON CONFLICT(itc_id) DO UPDATE SET"
                "  parent_itc = excluded.parent_itc,"
                "  cluster_idx = excluded.cluster_idx,"
                "  edit_id = excluded.edit_id,"
                "  las_path = excluded.las_path,"
                "  point_count = excluded.point_count,"
                "  status = 'active'",
                (int(itc_id), int(parent_itc), int(cluster_idx), int(edit_id), str(las_path), int(point_count))
            )
            self._conn.commit()

    def get_tree_edit(self, itc_id: int):
        """Lookup an active tree edit by Tree ID (ITC)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT itc_id, parent_itc, cluster_idx, edit_id, las_path, point_count, status "
                "FROM tree_edits_index WHERE itc_id = ? AND status = 'active'",
                (int(itc_id),)
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "itc_id": row[0],
                "parent_itc": row[1],
                "cluster_idx": row[2],
                "edit_id": row[3],
                "las_path": row[4],
                "point_count": row[5],
                "status": row[6]
            }

    def get_all_active_tree_edits(self) -> dict:
        """Return a mapping of all active edited Tree IDs -> edit info."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT itc_id, parent_itc, cluster_idx, edit_id, las_path, point_count, status "
                "FROM tree_edits_index WHERE status = 'active'"
            )
            rows = cur.fetchall()
            mapping = {}
            for r in rows:
                mapping[r[0]] = {
                    "itc_id": r[0],
                    "parent_itc": r[1],
                    "cluster_idx": r[2],
                    "edit_id": r[3],
                    "las_path": r[4],
                    "point_count": r[5],
                    "status": r[6]
                }
            return mapping

    def revert_tree_edits_by_edit_id(self, edit_id: int) -> int:
        """Mark tree edits for a given edit transaction as reverted."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE tree_edits_index SET status = 'reverted' WHERE edit_id = ?",
                (int(edit_id),)
            )
            self._conn.commit()
            return cur.rowcount

    def revert_tree_edits_by_parent_itc(self, parent_itc: int) -> int:
        """Mark tree edits for a given parent ITC as reverted."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE tree_edits_index SET status = 'reverted' WHERE parent_itc = ?",
                (int(parent_itc),)
            )
            self._conn.commit()
            return cur.rowcount

    # -------------------------------------------------------------------------
    # Staged (uncommitted) edit snapshot
    # -------------------------------------------------------------------------
    def save_staged_edits(self, edits: list) -> int:
        """Replace the staged-edit snapshot with `edits` (JSON per row)."""
        with self._lock:
            self._conn.execute("DELETE FROM staged_edits")
            self._conn.executemany(
                "INSERT INTO staged_edits(payload) VALUES(?)",
                [(json.dumps(e),) for e in edits]
            )
            self._conn.commit()
            return len(edits)

    def load_staged_edits(self) -> list:
        """Return the last staged-edit snapshot, in staging order."""
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                "SELECT payload FROM staged_edits ORDER BY idx"
            ).fetchall()
        out = []
        for (payload,) in rows:
            try:
                out.append(json.loads(payload))
            except (ValueError, TypeError):
                continue
        return out

    def close(self):
        with self._lock:
            self._conn.commit()
            self._conn.close()
