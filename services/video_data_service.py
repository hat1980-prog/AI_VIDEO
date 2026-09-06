import sqlite3
import shutil
import json
from pathlib import Path

from config import DATABASE_DIR, EMBEDDINGS_DIR, FACES_DIR, FRAMES_DIR, PERSON_LIBRARY_DIR
from database.db import get_connection
from services.person_library_service import invalidate_person_index, update_actor_embedding
from services.video_service import get_video_id, get_video_path


DATABASE_PATH = DATABASE_DIR / "metadata.db"


def _read_metadata(metadata_path):
    if not metadata_path.exists():
        return {}

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return metadata if isinstance(metadata, dict) else {}


def _is_same_source_video(info, video_name):
    if not isinstance(info, dict):
        return False

    stored_name = Path(info.get("video_name") or info.get("video_path", "")).name
    return stored_name.casefold() == video_name.casefold()


def _remove_actor_faces_for_video(video_name):
    removed_faces = 0
    affected_actors = set()

    for actor_dir in PERSON_LIBRARY_DIR.iterdir():
        if not actor_dir.is_dir():
            continue

        metadata_path = actor_dir / "faces_metadata.json"
        metadata = _read_metadata(metadata_path)
        if not metadata:
            continue

        face_names = [
            face_name
            for face_name, info in metadata.items()
            if _is_same_source_video(info, video_name)
        ]
        if not face_names:
            continue

        faces_dir = actor_dir / "faces"
        for face_name in face_names:
            face_path = faces_dir / face_name
            embedding_path = face_path.with_suffix(".npy")
            if face_path.exists():
                face_path.unlink()
                removed_faces += 1
            if embedding_path.exists():
                embedding_path.unlink()
            metadata.pop(face_name, None)

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        affected_actors.add(actor_dir.name)

    for actor_name in affected_actors:
        # Keep the actor's name and representative learning data even if this
        # was its only analysed video, so the new analysis can match it again.
        update_actor_embedding(actor_name)

    if affected_actors:
        invalidate_person_index()

    return removed_faces, len(affected_actors)


def _has_unknown_source(info):
    if not isinstance(info, dict):
        return True

    video_name = str(info.get("video_name") or "").strip()
    video_path = str(info.get("video_path") or "").strip()
    return not video_name or video_name == "元動画情報なし" or not video_path


def delete_unknown_source_faces():
    """Delete face images whose source-video metadata is unavailable.

    Actor identity folders and their representative image/embedding are retained
    so existing face learning can still be used by subsequent analyses.
    """
    extracted_faces = 0
    actor_faces = 0
    affected_actors = set()

    for face_dir in FACES_DIR.iterdir():
        if not face_dir.is_dir():
            continue

        metadata_path = face_dir / "faces_metadata.json"
        metadata = _read_metadata(metadata_path)
        face_names = [
            face_name for face_name, info in metadata.items()
            if _has_unknown_source(info)
        ]
        if not face_names:
            continue

        embedding_dir = EMBEDDINGS_DIR / face_dir.name
        for face_name in face_names:
            # Cluster folders are copies of the root face image.  Remove every
            # copy now; they will be rebuilt by the next clustering operation.
            for face_path in face_dir.rglob(face_name):
                if face_path.is_file():
                    face_path.unlink()
                    extracted_faces += 1
            embedding_path = embedding_dir / Path(face_name).with_suffix(".npy")
            if embedding_path.exists():
                embedding_path.unlink()
            metadata.pop(face_name, None)

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for state_file in ("assignments.json", "manual_exclusions.json"):
            state_path = face_dir / state_file
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(state, dict):
                state = {key: value for key, value in state.items() if key not in face_names}
            elif isinstance(state, list):
                state = [item for item in state if item not in face_names]
            else:
                continue
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    for actor_dir in PERSON_LIBRARY_DIR.iterdir():
        if not actor_dir.is_dir():
            continue

        metadata_path = actor_dir / "faces_metadata.json"
        metadata = _read_metadata(metadata_path)
        face_names = [
            face_name for face_name, info in metadata.items()
            if _has_unknown_source(info)
        ]
        if not face_names:
            continue

        faces_dir = actor_dir / "faces"
        for face_name in face_names:
            face_path = faces_dir / face_name
            embedding_path = face_path.with_suffix(".npy")
            if face_path.exists():
                face_path.unlink()
                actor_faces += 1
            if embedding_path.exists():
                embedding_path.unlink()
            metadata.pop(face_name, None)

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        affected_actors.add(actor_dir.name)

    for actor_name in affected_actors:
        update_actor_embedding(actor_name)
    if affected_actors:
        invalidate_person_index()

    return (
        f"元動画情報なしの顔画像を削除: 抽出顔{extracted_faces}枚、"
        f"出演者ライブラリ{actor_faces}枚（{len(affected_actors)}人）。"
        "出演者名・代表画像・学習用代表Embeddingは保持しました"
    )


def clear_video_data_for_reanalysis(video):
    """Remove old results for a video before analysing it again.

    Matches by filename as well as path, handling a video that has been moved
    since an earlier analysis.  Actor identity/representative data is retained.
    """
    video_path = get_video_path(video)
    if video_path is None:
        return "元動画が見つかりません"

    video_name = video_path.name
    video_ids = set()
    for face_dir in FACES_DIR.iterdir():
        if not face_dir.is_dir():
            continue

        metadata = _read_metadata(face_dir / "faces_metadata.json")
        if any(_is_same_source_video(info, video_name) for info in metadata.values()):
            video_ids.add(face_dir.name)

    # Frame and embedding directories do not store source-video metadata, so
    # remove their companions for every matched face-data directory.
    current_video_id = get_video_id(video)
    video_ids.add(current_video_id)
    directories = [
        base_dir / video_id
        for video_id in video_ids
        for base_dir in (FRAMES_DIR, FACES_DIR, EMBEDDINGS_DIR)
        if (base_dir / video_id).exists()
    ]

    file_count = sum(
        sum(1 for path in directory.rglob("*") if path.is_file())
        for directory in directories
    )
    _delete_database_records(directories)

    for directory in directories:
        shutil.rmtree(directory)

    actor_faces, actor_count = _remove_actor_faces_for_video(video_name)
    return (
        f"再解析前のデータを削除: 解析ファイル{file_count}件（{len(directories)}フォルダ）、"
        f"出演者ライブラリの顔画像{actor_faces}枚（{actor_count}人）。"
        "出演者名・代表画像・学習用代表Embeddingは保持しました"
    )


def _delete_database_records(paths):
    if not DATABASE_PATH.exists():
        return

    conn = get_connection()

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
