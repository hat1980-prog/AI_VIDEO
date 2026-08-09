import json
from pathlib import Path

from config import FACES_DIR, PERSON_LIBRARY_DIR


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
