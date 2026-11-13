import json
import os
from pathlib import Path

from config import FACES_DIR, PERSON_LIBRARY_DIR
from database.db import get_connection


def _same_path(first, second):
    """Compare saved Windows paths without requiring the old path to exist."""
    if not first or not second:
        return False
    return os.path.normcase(os.path.normpath(str(first))) == os.path.normcase(
        os.path.normpath(str(second))
    )


def _normalize_metadata_file(metadata_path, video_name, video_path):
    if not metadata_path.exists():
        return 0

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0

    updated = 0

    for info in metadata.values():
        existing_name = Path(info.get("video_name", "")).name

        if existing_name.casefold() != video_name.casefold():
            continue

        if info.get("video_path") != video_path or info.get("video_name") != video_name:
            info["video_name"] = video_name
            info["video_path"] = video_path
            updated += 1

    if updated:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return updated


def _update_metadata_for_renamed_video(
    metadata_path, video_name, video_path, video_id, previous_paths, force=False,
):
    """Update only entries proven to belong to the renamed file identity."""
    if not metadata_path.exists():
        return 0

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(metadata, dict):
        return 0

    updated = 0
    for info in metadata.values():
        if not isinstance(info, dict):
            continue
        # New records carry video_id.  Earlier records are matched by their
        # exact former path, never just a filename, to avoid changing a
        # different work with the same title.
        belongs_to_video = (
            force
            or info.get("video_id") == video_id
            or any(
                _same_path(info.get("video_path"), previous_path)
                for previous_path in previous_paths
            )
        )
        if not belongs_to_video:
            continue
        if (
            info.get("video_name") != video_name
            or info.get("video_path") != video_path
            or info.get("video_id") != video_id
        ):
            info["video_name"] = video_name
            info["video_path"] = video_path
            info["video_id"] = video_id
            updated += 1

    if updated:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return updated


def _metadata_source_paths(metadata_path):
    """Read historical source paths before normalizing an extracted-face set."""
    if not metadata_path.exists():
        return set()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {
        str(info["video_path"])
        for info in metadata.values()
        if isinstance(info, dict) and info.get("video_path")
    }


def normalize_renamed_video_identity(video_path, video_id, previous_path=None):
    """Follow a rename/move detected by the stable filesystem identity.

    The extracted-face directory is directly addressed by video_id.  Actor
    folders contain copied faces, so their metadata is updated only when the
    old absolute path (or a newer stored video_id) proves the association.
    """
    video_path = Path(video_path).resolve()
    video_name = video_path.name
    normalized_path = str(video_path)
    actor_faces = 0

    face_dir = FACES_DIR / video_id
    # A previous implementation could update the identity index before the
    # copied actor-face metadata.  Read the extracted-face folder first so a
    # later "ファイル情報を更新" can repair such already-recorded renames too.
    previous_paths = _metadata_source_paths(face_dir / "faces_metadata.json")
    if previous_path:
        previous_paths.add(str(previous_path))
    extracted_faces = _update_metadata_for_renamed_video(
        face_dir / "faces_metadata.json",
        video_name,
        normalized_path,
        video_id,
        previous_paths,
        force=True,
    )

    for actor_dir in PERSON_LIBRARY_DIR.iterdir():
        if actor_dir.is_dir():
            actor_faces += _update_metadata_for_renamed_video(
                actor_dir / "faces_metadata.json",
                video_name,
                normalized_path,
                video_id,
                previous_paths,
            )

    # The legacy SQLite table is not the face-library source of truth, but
    # keep its display record in sync when a matching prior path exists.
    try:
        conn = get_connection()
        try:
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "videos" in tables:
                conn.execute(
                    "UPDATE videos SET filename = ?, filepath = ? WHERE filepath = ?",
                    (video_name, normalized_path, str(previous_path)),
                )
                conn.commit()
        finally:
            conn.close()
    except Exception:
        # A missing/legacy database must not prevent a scan from completing.
        pass

    return actor_faces, extracted_faces


def normalize_source_video_paths(video_path):
    video_path = Path(video_path).resolve()
    video_name = video_path.name
    normalized_path = str(video_path)
    actor_faces = 0
    extracted_faces = 0

    for actor_dir in PERSON_LIBRARY_DIR.iterdir():
        if actor_dir.is_dir():
            actor_faces += _normalize_metadata_file(
                actor_dir / "faces_metadata.json", video_name, normalized_path
            )

    for face_dir in FACES_DIR.iterdir():
        if face_dir.is_dir():
            extracted_faces += _normalize_metadata_file(
                face_dir / "faces_metadata.json", video_name, normalized_path
            )

    return actor_faces, extracted_faces
