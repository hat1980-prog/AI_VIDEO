import sqlite3

from config import DATABASE_DIR


DATABASE_PATH = DATABASE_DIR / "metadata.db"


def get_database_tables():
    if not DATABASE_PATH.exists():
        return []

    conn = sqlite3.connect(DATABASE_PATH)

    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()

    return [row[0] for row in rows]


def get_table_data(table_name, limit=500):
    tables = get_database_tables()

    if table_name not in tables:
        return [], [], "表示できるテーブルを選択してください"

    conn = sqlite3.connect(DATABASE_PATH)

    try:
        cursor = conn.execute(f'PRAGMA table_info("{table_name}")')
        headers = [row[1] for row in cursor.fetchall()]
        rows = conn.execute(f'SELECT * FROM "{table_name}" LIMIT ?', (limit,)).fetchall()
    finally:
        conn.close()

    return headers, rows, f"{table_name}: {len(rows)}件を表示"
