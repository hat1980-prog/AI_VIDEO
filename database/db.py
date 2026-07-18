import sqlite3

from config import DATABASE_DIR

DB_PATH = DATABASE_DIR / "metadata.db"


def _ensure_indexes(conn):
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    if "faces" not in tables:
        return

    columns = {row[1] for row in conn.execute("PRAGMA table_info(faces)")}

    for column in ("video_id", "cluster_id", "image_path", "face_path"):
        if column in columns:
            conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_faces_{column}" ON faces("{column}")')


def get_connection():

    conn = sqlite3.connect(DB_PATH, timeout=30)

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA cache_size = -20000")
    _ensure_indexes(conn)

    return conn
