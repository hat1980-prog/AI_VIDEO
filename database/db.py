import sqlite3

from config import DATABASE_DIR

DB_PATH = DATABASE_DIR / "metadata.db"


def _ensure_indexes(conn):
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    if "faces" in tables:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(faces)")}

        for column in ("video_id", "cluster_id", "image_path", "face_path"):
            if column in columns:
                conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_faces_{column}" ON faces("{column}")')

        # The viewer and cleanup paths commonly filter by video then cluster.
        # A composite index avoids repeatedly intersecting two single-column scans.
        if {"video_id", "cluster_id"} <= columns:
            conn.execute(
                'CREATE INDEX IF NOT EXISTS "idx_faces_video_cluster" '
                'ON faces("video_id", "cluster_id")'
            )

    if "excluded_faces" in tables:
        excluded_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(excluded_faces)")
        }
        if "source_face_path" in excluded_columns:
            conn.execute(
                'CREATE INDEX IF NOT EXISTS "idx_excluded_faces_source_path" '
                'ON excluded_faces("source_face_path")'
            )


def get_connection():

    conn = sqlite3.connect(DB_PATH, timeout=30)

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA cache_size = -20000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA wal_autocheckpoint = 1000")
    _ensure_indexes(conn)

    return conn


def optimize_database():
    """Refresh SQLite query statistics without rebuilding or deleting user data."""
    conn = get_connection()
    try:
        conn.execute("ANALYZE")
        conn.execute("PRAGMA optimize")
        conn.commit()
    finally:
        conn.close()
