import sqlite3
import shutil

from config import (
    ACTORS_DIR,
    DATABASE_DIR,
    EXCLUDED_FACES_DIR,
    PERSON_LIBRARY_DIR,
    PERSON_LIBRARY_SETTINGS_PATH,
)


DATABASE_PATH = DATABASE_DIR / "metadata.db"


def reset_database():
    for path in (
        DATABASE_PATH,
        DATABASE_PATH.with_name(f"{DATABASE_PATH.name}-wal"),
        DATABASE_PATH.with_name(f"{DATABASE_PATH.name}-shm"),
    ):
        if path.exists():
            path.unlink()

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            transcript TEXT,
            tags TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            cluster_id INTEGER,
            image_path TEXT
        )
        """
    )

    conn.commit()
    conn.close()

    if EXCLUDED_FACES_DIR.exists():
        shutil.rmtree(EXCLUDED_FACES_DIR)
    EXCLUDED_FACES_DIR.mkdir(parents=True, exist_ok=True)

    return "データベースを初期化しました"


def reset_actor_library():
    for directory in (PERSON_LIBRARY_DIR, ACTORS_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    if PERSON_LIBRARY_SETTINGS_PATH.exists():
        PERSON_LIBRARY_SETTINGS_PATH.unlink()

    return "出演者ライブラリを初期化しました"
