import json
from pathlib import Path
import shutil

import numpy as np

from config import EMBEDDINGS_DIR, FACES_DIR, PERSON_LIBRARY_DIR
from services.actor_service import load_actor_list
from services.person_library_service import load_all_persons, update_actor_embedding
from services.video_service import get_video_id


def _paths(video):
    video_id = get_video_id(video)

    if video_id is None:
        return None, None

    return FACES_DIR / video_id, EMBEDDINGS_DIR / video_id


def _load_json(path, default):
    if not path.exists():
        return default

    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def get_manual_targets(video):
    face_dir, _ = _paths(video)

    if face_dir is None or not face_dir.exists():
        return []

    targets = [
        (f"人物一覧: {folder.name}", f"cluster:{folder.name}")
        for folder in sorted(face_dir.glob("Person_*"))
    ]
    targets.extend(
        (f"出演者ライブラリ: {actor_name}", f"actor:{actor_name}")
        for actor_name in load_actor_list()
    )
    return targets


def get_face_similarity_candidates(video, selected_face_path, limit=5):
    """Return the closest registered actors for a selected extracted face."""
    face_dir, embedding_dir = _paths(video)
    if face_dir is None:
        return []
    filename = _validate_selected_face(face_dir, selected_face_path)
    if filename is None:
        return []
    embedding_path = embedding_dir / f"{Path(filename).stem}.npy"
    if not embedding_path.exists():
        return []

    embedding = np.load(embedding_path)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return []
    embedding = embedding / norm
    candidates = []
    for person in load_all_persons():
        similarity = float(np.dot(person["embedding"], embedding))
        candidates.append((person["name"], similarity))
    return sorted(candidates, key=lambda candidate: candidate[1], reverse=True)[:limit]


def _validate_selected_face(face_dir, selected_face_path):
    if not selected_face_path:
        return None

    selected_path = Path(selected_face_path)

    try:
        selected_path.resolve().relative_to(face_dir.resolve())
    except ValueError:
        return None

    return selected_path.name


def _remove_from_groups(face_dir, filename):
    for pattern in ("Actor_*", "Person_*"):
        for folder in face_dir.glob(pattern):
            path = folder / filename

            if path.exists():
                path.unlink()

    unknown_path = face_dir / "Unknown" / filename

    if unknown_path.exists():
        unknown_path.unlink()


def exclude_from_classification(video, selected_face_path):
    face_dir, _ = _paths(video)

    if face_dir is None:
        return "動画ファイルが見つかりません"

    filename = _validate_selected_face(face_dir, selected_face_path)

    if filename is None:
        return "分類から除外する顔画像を選択してください"

    source_path = face_dir / filename

    if not source_path.exists():
        return "元の顔画像が見つかりません"

    _remove_from_groups(face_dir, filename)
    unclassified_dir = face_dir / "Unclassified"
    unclassified_dir.mkdir(exist_ok=True)
    shutil.copy2(source_path, unclassified_dir / filename)

    assignments_path = face_dir / "assignments.json"
    assignments = _load_json(assignments_path, {})
    assignments.pop(filename, None)
    _save_json(assignments_path, assignments)

    exclusions_path = face_dir / "manual_exclusions.json"
    exclusions = set(_load_json(exclusions_path, []))
    exclusions.add(filename)
    _save_json(exclusions_path, sorted(exclusions))

    return "顔画像を未分類へ移動しました。手動で再分類するまで自動分類から除外されます"


def reclassify_face(video, selected_face_path, target):
    face_dir, embedding_dir = _paths(video)

    if face_dir is None:
        return "動画ファイルが見つかりません"

    filename = _validate_selected_face(face_dir, selected_face_path)

    if filename is None or not target:
        return "顔画像と手動分類先を選択してください"

    source_path = face_dir / filename
    embedding_path = embedding_dir / f"{Path(filename).stem}.npy"

    if not source_path.exists() or not embedding_path.exists():
        return "顔画像またはEmbeddingが見つかりません"

    _remove_from_groups(face_dir, filename)
    unclassified_path = face_dir / "Unclassified" / filename

    if unclassified_path.exists():
        unclassified_path.unlink()

    assignments_path = face_dir / "assignments.json"
    assignments = _load_json(assignments_path, {})
    exclusions_path = face_dir / "manual_exclusions.json"
    exclusions = set(_load_json(exclusions_path, []))
    exclusions.discard(filename)

    target_type, _, target_name = target.partition(":")

    if target_type == "cluster" and target_name.startswith("Person_"):
        destination_dir = face_dir / target_name
        destination_dir.mkdir(exist_ok=True)
        shutil.copy2(source_path, destination_dir / filename)
        assignments.pop(filename, None)
        message = f"{filename} を {target_name} へ再分類しました"
    elif target_type == "actor" and target_name in load_actor_list():
        destination_dir = face_dir / f"Actor_{target_name}"
        destination_dir.mkdir(exist_ok=True)
        shutil.copy2(source_path, destination_dir / filename)
        assignments[filename] = target_name

        actor_dir = PERSON_LIBRARY_DIR / target_name
        actor_faces_dir = actor_dir / "faces"
        actor_faces_dir.mkdir(parents=True, exist_ok=True)
        actor_face_path = actor_faces_dir / filename
        index = 1

        while actor_face_path.exists():
            actor_face_path = actor_faces_dir / f"{Path(filename).stem}_{index}.jpg"
            index += 1

        shutil.copy2(source_path, actor_face_path)
        shutil.copy2(embedding_path, actor_face_path.with_suffix(".npy"))

        face_metadata = _load_json(face_dir / "faces_metadata.json", {})
        actor_metadata_path = actor_dir / "faces_metadata.json"
        actor_metadata = _load_json(actor_metadata_path, {})
        actor_metadata[actor_face_path.name] = face_metadata.get(
            filename,
            {"video_name": "元動画情報なし", "video_path": ""},
        )
        _save_json(actor_metadata_path, actor_metadata)
        update_actor_embedding(target_name)
        message = f"{filename} を出演者 {target_name} へ再分類しました"
    else:
        return "指定した手動分類先は利用できません"

    _save_json(assignments_path, assignments)
    _save_json(exclusions_path, sorted(exclusions))
    return message
