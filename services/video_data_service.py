import sqlite3
import shutil

from config import DATABASE_DIR, EMBEDDINGS_DIR, FACES_DIR, FRAMES_DIR
from services.video_service import get_video_id


DATABASE_PATH = DATABASE_DIR / "metadata.db"


def _delete_database_records(paths):
    if not DATABASE_PATH.exists():
        return

    conn = sqlite3.connect(DATABASE_PATH)

    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        if "faces" not in tables:
            return

        columns = {row[1] for row in conn.execute("PRAGMA table_info(faces)")}

        for column in {"image_path", "face_path"} & columns:
            for path in paths:
                conn.execute(f'DELETE FROM faces WHERE "{column}" LIKE ?', (f"{path}%",))

        conn.commit()
    finally:
        conn.close()


def delete_video_analysis_data(video):
    video_id = get_video_id(video)

    if video_id is None:
        return "元動画を選択してください"

    directories = [
        FRAMES_DIR / video_id,
        FACES_DIR / video_id,
        EMBEDDINGS_DIR / video_id,
    ]
    file_count = sum(
        sum(1 for path in directory.rglob("*") if path.is_file())
        for directory in directories
        if directory.exists()
    )

    _delete_database_records(directories)

    removed = 0

    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)
            removed += 1

    if not removed:
        return "選択した元動画に対応する解析データはありません"

    return (
        f"{video_id} の解析データを削除しました（{file_count}ファイル）。"
        "出演者ライブラリと除外人物の学習データは保持されています"
    )
