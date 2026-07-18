from hashlib import sha1
from pathlib import Path
import sqlite3

import numpy as np

from config import (
    DATABASE_DIR,
    EMBEDDINGS_DIR,
    EXCLUDED_FACE_SIMILARITY_THRESHOLD,
    EXCLUDED_FACES_DIR,
    FACES_DIR,
)
from services.video_service import get_video_id


DATABASE_PATH = DATABASE_DIR / "metadata.db"


def _cosine_similarity(first_embedding, second_embedding):
    denominator = np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding)

    if denominator == 0:
        return -1.0

    return float(np.dot(first_embedding, second_embedding) / denominator)


def is_excluded_face(embedding):
    for excluded_embedding_path in EXCLUDED_FACES_DIR.glob("*.npy"):
        excluded_embedding = np.load(excluded_embedding_path)

        if _cosine_similarity(embedding, excluded_embedding) >= EXCLUDED_FACE_SIMILARITY_THRESHOLD:
            return True

    return False


def _delete_database_records(face_paths):
    if not DATABASE_PATH.exists():
        return

    conn = sqlite3.connect(DATABASE_PATH)

    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        if "faces" not in tables:
            return

        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(faces)").fetchall()
        }

        for column in {"image_path", "face_path"} & columns:
            conn.executemany(
                f'DELETE FROM faces WHERE "{column}" = ?',
                [(str(path),) for path in face_paths],
            )

        conn.commit()
    finally:
        conn.close()


def _record_excluded_face(source_face_path, excluded_embedding_path):
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS excluded_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_face_path TEXT NOT NULL,
                embedding_path TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO excluded_faces(source_face_path, embedding_path) VALUES(?, ?)",
            (str(source_face_path), str(excluded_embedding_path)),
        )
        conn.commit()
    finally:
        conn.close()


def exclude_face(video, selected_face_path):
    video_id = get_video_id(video)

    if video_id is None or not selected_face_path:
        return "除外する顔画像を選択してください"

    face_dir = FACES_DIR / video_id
    embedding_dir = EMBEDDINGS_DIR / video_id
    selected_path = Path(selected_face_path)

    try:
        selected_path.resolve().relative_to(face_dir.resolve())
    except ValueError:
        return "選択された顔画像は対象動画のものではありません"

    source_face_path = face_dir / selected_path.name
    embedding_path = embedding_dir / f"{selected_path.stem}.npy"

    if not embedding_path.exists():
        return "対応するEmbeddingが見つかりません"

    embedding = np.load(embedding_path)
    identifier = sha1(embedding.tobytes()).hexdigest()
    excluded_embedding_path = EXCLUDED_FACES_DIR / f"{identifier}.npy"
    np.save(excluded_embedding_path, embedding)
    _record_excluded_face(source_face_path, excluded_embedding_path)

    deleted_paths = []

    for face_path in face_dir.rglob(selected_path.name):
        face_path.unlink()
        deleted_paths.append(face_path)

    if source_face_path.exists():
        source_face_path.unlink()
        deleted_paths.append(source_face_path)

    embedding_path.unlink()
    _delete_database_records(deleted_paths)

    return "顔画像を除外しました。以降、この人物に一致する顔は無視されます"
