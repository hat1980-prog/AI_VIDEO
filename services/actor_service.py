from pathlib import Path
import json
import re
import shutil

import numpy as np

from config import PERSON_LIBRARY_DIR, RECLASSIFICATION_FACE_EXEMPLARS
from services.person_library_service import (
    calibrate_similarity_threshold,
    get_similarity_threshold,
    invalidate_person_index,
)


_ACTOR_RECORDS_CACHE = {"fingerprint": None, "records": {}}


def _actor_dir(actor_name):
    if not actor_name:
        return None

    actor_dir = PERSON_LIBRARY_DIR / actor_name

    if actor_dir.parent != PERSON_LIBRARY_DIR or not actor_dir.is_dir():
        return None

    return actor_dir


def _representative_image(actor_dir):
    representative = actor_dir / "representative.jpg"

    if representative.exists():
        return representative

    faces_dir = actor_dir / "faces"
    images = sorted(faces_dir.glob("*.jpg")) if faces_dir.exists() else []

    return images[0] if images else None


def _update_actor_embedding(actor_name):
    actor_dir = _actor_dir(actor_name)

    if actor_dir is None:
        return False

    embeddings = []

    for image in sorted((actor_dir / "faces").glob("*.jpg")):
        embedding_path = image.with_suffix(".npy")

        if embedding_path.exists():
            embeddings.append(np.load(embedding_path))

    if not embeddings:
        return False

    mean_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_embedding)

    if norm == 0:
        return False

    np.save(actor_dir / "representative.npy", mean_embedding / norm)
    invalidate_person_index()
    return True


def _similarity(first_embedding, second_embedding):
    denominator = np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding)

    if denominator == 0:
        return None

    return float(np.dot(first_embedding, second_embedding) / denominator)


def _is_generated_actor_name(actor_name):
    return re.fullmatch(r"Person\d+", actor_name) is not None


def _merge_actor_files(source_dir, target_dir):
    source_faces = source_dir / "faces"
    target_faces = target_dir / "faces"
    target_faces.mkdir(parents=True, exist_ok=True)
    source_metadata_path = source_dir / "faces_metadata.json"
    target_metadata_path = target_dir / "faces_metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8")) if source_metadata_path.exists() else {}
    target_metadata = json.loads(target_metadata_path.read_text(encoding="utf-8")) if target_metadata_path.exists() else {}

    moved = 0

    if source_faces.exists():
        for face_path in sorted(source_faces.glob("*.jpg")):
            destination = target_faces / face_path.name
            index = 1

            while destination.exists():
                destination = target_faces / f"{face_path.stem}_{index}{face_path.suffix}"
                index += 1

            shutil.move(str(face_path), str(destination))
            embedding = face_path.with_suffix(".npy")

            if embedding.exists():
                shutil.move(str(embedding), str(destination.with_suffix(".npy")))

            target_metadata[destination.name] = source_metadata.get(
                face_path.name,
                {"video_name": "元動画情報なし", "video_path": ""},
            )

            moved += 1

    source_name = source_dir.name
    target_name = target_dir.name
    shutil.rmtree(source_dir)
    target_metadata_path.write_text(
        json.dumps(target_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    _update_actor_embedding(target_name)

    return source_name, moved


def _source_video_names(*actor_dirs):
    video_names = set()
    for actor_dir in actor_dirs:
        metadata, _ = _load_face_metadata(actor_dir)
        for info in metadata.values():
            if not isinstance(info, dict):
                continue
            video_name = Path(info.get("video_name") or info.get("video_path", "")).name
            if video_name and video_name != "元動画情報なし":
                video_names.add(video_name.casefold())
    return video_names


def _reclassify_related_generated_faces(target_dir, video_names, threshold):
    """Move matching generated-person faces from the same videos to target."""
    if not video_names:
        return 0, 0

    target_embedding_path = target_dir / "representative.npy"
    if not target_embedding_path.exists():
        return 0, 0
    target_embedding = np.load(target_embedding_path)
    target_norm = np.linalg.norm(target_embedding)
    if target_norm == 0:
        return 0, 0
    target_embedding = target_embedding / target_norm

    target_faces = target_dir / "faces"
    target_metadata, target_metadata_path = _load_face_metadata(target_dir)
    moved = 0
    checked = 0
    affected_sources = []

    for source_dir in list(PERSON_LIBRARY_DIR.iterdir()):
        if (
            not source_dir.is_dir()
            or source_dir == target_dir
            or not _is_generated_actor_name(source_dir.name)
        ):
            continue

        source_metadata, source_metadata_path = _load_face_metadata(source_dir)
        source_faces = source_dir / "faces"
        changed = False
        for face_path in list(source_faces.glob("*.jpg")):
            info = source_metadata.get(face_path.name, {})
            video_name = Path(info.get("video_name") or info.get("video_path", "")).name
            if video_name.casefold() not in video_names:
                continue
            embedding_path = face_path.with_suffix(".npy")
            if not embedding_path.exists():
                continue
            checked += 1
            embedding = np.load(embedding_path)
            similarity = _similarity(embedding, target_embedding)
            if similarity is None or similarity < threshold:
                continue

            destination = target_faces / face_path.name
            suffix = 1
            while destination.exists():
                destination = target_faces / f"{face_path.stem}_{suffix}{face_path.suffix}"
                suffix += 1
            shutil.move(str(face_path), str(destination))
            shutil.move(str(embedding_path), str(destination.with_suffix(".npy")))
            target_metadata[destination.name] = source_metadata.pop(
                face_path.name,
                {"video_name": "元動画情報なし", "video_path": ""},
            )
            moved += 1
            changed = True

        if changed:
            source_metadata_path.write_text(
                json.dumps(source_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            affected_sources.append(source_dir)

    if moved:
        target_metadata_path.write_text(
            json.dumps(target_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for source_dir in affected_sources:
            if source_dir.exists():
                _refresh_actor_after_face_change(source_dir)
        _refresh_actor_after_face_change(target_dir)

    return moved, checked


def reclassify_actors(threshold=None, progress_callback=None, should_cancel=None):
    """Merge generated Person entries using a bounded, one-pass similarity plan."""
    threshold = get_similarity_threshold() if threshold is None else threshold
    actor_names = []
    embeddings = []

    for actor_name in load_actor_list():
        embedding_path = _actor_dir(actor_name) / "representative.npy"
        if not embedding_path.exists():
            continue
        embedding = np.load(embedding_path)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            actor_names.append(actor_name)
            embeddings.append(embedding / norm)

    if len(actor_names) < 2:
        return [], False

    matrix = np.stack(embeddings)
    generated_indexes = [
        index for index, name in enumerate(actor_names) if _is_generated_actor_name(name)
    ]
    manual_indexes = [
        index for index, name in enumerate(actor_names) if not _is_generated_actor_name(name)
    ]
    manual_exemplars = []
    manual_exemplar_owners = []
    for owner_index in manual_indexes:
        actor_dir = _actor_dir(actor_names[owner_index])
        embedding_paths = sorted((actor_dir / "faces").glob("*.npy")) if actor_dir else []
        if len(embedding_paths) > RECLASSIFICATION_FACE_EXEMPLARS:
            indexes = np.linspace(
                0, len(embedding_paths) - 1, RECLASSIFICATION_FACE_EXEMPLARS, dtype=int
            )
            embedding_paths = [embedding_paths[index] for index in indexes]
        for embedding_path in embedding_paths:
            embedding = np.load(embedding_path)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                manual_exemplars.append(embedding / norm)
                manual_exemplar_owners.append(owner_index)

        # Keep the representative as a fallback for actors without face files.
        if not embedding_paths:
            manual_exemplars.append(matrix[owner_index])
            manual_exemplar_owners.append(owner_index)
    manual_exemplar_matrix = np.stack(manual_exemplars) if manual_exemplars else None
    parent = list(range(len(actor_names)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    chunk_size = 32
    total_candidates = len(generated_indexes)
    for offset in range(0, total_candidates, chunk_size):
        if should_cancel and should_cancel():
            return [], True
        indexes = generated_indexes[offset:offset + chunk_size]
        if manual_exemplar_matrix is not None:
            similarities = manual_exemplar_matrix @ matrix[indexes].T
            for column, source_index in enumerate(indexes):
                best_index = int(np.argmax(similarities[:, column]))
                if similarities[best_index, column] >= threshold:
                    union(source_index, manual_exemplar_owners[best_index])
        else:
            similarities = matrix @ matrix[indexes].T
            for column, source_index in enumerate(indexes):
                scores = similarities[:, column].copy()
                scores[source_index] = -np.inf
                candidate_index = int(np.argmax(scores))
                if scores[candidate_index] >= threshold:
                    union(source_index, candidate_index)
        if progress_callback:
            progress_callback("照合", min(offset + len(indexes), total_candidates), total_candidates)

    components = {}
    for index in range(len(actor_names)):
        components.setdefault(find(index), []).append(index)

    merge_plan = []
    for indexes in components.values():
        generated = [index for index in indexes if _is_generated_actor_name(actor_names[index])]
        manual = [index for index in indexes if not _is_generated_actor_name(actor_names[index])]
        if not generated:
            continue
        # Never merge independently named actors.  Prefer the single manual
        # actor in a component; otherwise retain the oldest generated name.
        if len(manual) > 1:
            continue
        target_index = manual[0] if manual else min(generated, key=lambda index: actor_names[index])
        for source_index in generated:
            if source_index != target_index:
                merge_plan.append((actor_names[source_index], actor_names[target_index]))

    merged = []
    total_merges = len(merge_plan)
    for index, (source_name, target_name) in enumerate(merge_plan, start=1):
        if should_cancel and should_cancel():
            return merged, True
        source_dir, target_dir = _actor_dir(source_name), _actor_dir(target_name)
        if source_dir is None or target_dir is None or source_dir == target_dir:
            continue
        _merge_actor_files(source_dir, target_dir)
        merged.append(source_name)
        if progress_callback:
            progress_callback("統合", index, total_merges)

    return merged, False


def _actor_face_paths(actor_dir, source_video_path=None):
    faces_dir = actor_dir / "faces"
    faces = sorted(faces_dir.glob("*.jpg")) if faces_dir.exists() else []

    if not source_video_path:
        return faces

    metadata, _ = _load_face_metadata(actor_dir)
    source_video_name = Path(source_video_path).name.casefold()
    return [
        face for face in faces
        if Path(
            metadata.get(face.name, {}).get("video_name")
            or metadata.get(face.name, {}).get("video_path", "")
        ).name.casefold() == source_video_name
    ]


def get_actor_video_choices(actor_name_filter=None):
    records = {}
    actor_video_names = set()

    for actor_dir in PERSON_LIBRARY_DIR.iterdir():
        if not actor_dir.is_dir():
            continue

        metadata, _ = _load_face_metadata(actor_dir)

        for face_name, info in metadata.items():
            video_path = info.get("video_path")

            if not video_path:
                continue

            video_name = Path(info.get("video_name") or video_path).name
            key = video_name.casefold()

            if actor_name_filter and actor_dir.name == actor_name_filter:
                actor_video_names.add(key)

            face_path = actor_dir / "faces" / face_name
            modified_at = face_path.stat().st_mtime_ns if face_path.exists() else 0
            current = records.get(key)

            if current is None or modified_at > current[2]:
                records[key] = (video_name, video_path, modified_at)

    choices = [
        (video_name, video_path)
        for video_name, video_path, _ in sorted(
            records.values(), key=lambda record: (record[0].casefold(), record[1])
        )
    ]
    if not actor_name_filter:
        return choices

    # Keep dropdown values stable when switching between video and actor filters.
    # A selected actor may only retain an old path for a video that was later
    # re-analysed elsewhere, but the UI must keep using the global canonical path.
    return [choice for choice in choices if choice[0].casefold() in actor_video_names]


def _actor_records_fingerprint():
    records = []
    for actor_dir in sorted(PERSON_LIBRARY_DIR.iterdir(), key=lambda path: path.name.casefold()):
        if not actor_dir.is_dir():
            continue
        faces_dir = actor_dir / "faces"
        metadata_path = actor_dir / "faces_metadata.json"
        representative = actor_dir / "representative.jpg"
        records.append(
            (
                actor_dir.name,
                faces_dir.stat().st_mtime_ns if faces_dir.exists() else 0,
                metadata_path.stat().st_mtime_ns if metadata_path.exists() else 0,
                representative.stat().st_mtime_ns if representative.exists() else 0,
            )
        )
    return tuple(records)


def _actor_records(source_video_path=None):
    fingerprint = _actor_records_fingerprint()
    source_key = Path(source_video_path).name.casefold() if source_video_path else None

    if _ACTOR_RECORDS_CACHE["fingerprint"] != fingerprint:
        _ACTOR_RECORDS_CACHE["fingerprint"] = fingerprint
        _ACTOR_RECORDS_CACHE["records"] = {}

    cached = _ACTOR_RECORDS_CACHE["records"].get(source_key)
    if cached is not None:
        return list(cached)

    records = []

    for actor_dir in PERSON_LIBRARY_DIR.iterdir():
        if not actor_dir.is_dir() or not _representative_image(actor_dir):
            continue

        faces_dir = actor_dir / "faces"
        faces = sorted(faces_dir.glob("*.jpg")) if faces_dir.exists() else []
        metadata, _ = _load_face_metadata(actor_dir)

        if source_key:
            faces = [
                face for face in faces
                if Path(
                    metadata.get(face.name, {}).get("video_name")
                    or metadata.get(face.name, {}).get("video_path", "")
                ).name.casefold() == source_key
            ]

        face_count = len(faces)

        if source_video_path and face_count == 0:
            continue

        # Count unique source titles, not face images.  Filenames are used as
        # the canonical identity throughout the library so moved videos are
        # still treated as the same work.
        video_names = {
            Path(info.get("video_name") or info.get("video_path", "")).name.casefold()
            for info in metadata.values()
            if isinstance(info, dict)
            and Path(info.get("video_name") or info.get("video_path", "")).name
            and Path(info.get("video_name") or info.get("video_path", "")).name != "元動画情報なし"
        }
        records.append((actor_dir.name, face_count, len(video_names)))

    _ACTOR_RECORDS_CACHE["records"][source_key] = tuple(records)
    return records


def load_actor_list(
    sort_by="name", sort_order="asc", source_video_path=None, actor_name_filter=None
):
    records = _actor_records(source_video_path)

    if actor_name_filter and sort_by != "similarity":
        records = [record for record in records if record[0] == actor_name_filter]

    reverse = sort_order == "desc"

    if sort_by == "similarity" and actor_name_filter:
        reference_path = _actor_dir(actor_name_filter) / "representative.npy"
        similarities = {}
        if reference_path.exists():
            reference = np.load(reference_path)
            for actor_name, _, _ in records:
                embedding_path = _actor_dir(actor_name) / "representative.npy"
                if embedding_path.exists():
                    similarity = _similarity(reference, np.load(embedding_path))
                    similarities[actor_name] = similarity if similarity is not None else -1.0
        records.sort(
            key=lambda record: (similarities.get(record[0], -1.0), record[0].casefold()),
            reverse=True,
        )
    elif sort_by == "count":
        records.sort(key=lambda record: (record[1], record[0].casefold()), reverse=reverse)
    elif sort_by == "works":
        records.sort(key=lambda record: (record[2], record[0].casefold()), reverse=reverse)
    else:
        records.sort(key=lambda record: record[0].casefold(), reverse=reverse)

    return [name for name, _, _ in records]


def load_actor_gallery(
    sort_by="name", sort_order="asc", source_video_path=None, actor_name_filter=None
):
    gallery = []

    for actor_name in load_actor_list(
        sort_by, sort_order, source_video_path, actor_name_filter
    ):
        actor_dir = PERSON_LIBRARY_DIR / actor_name
        filtered_faces = _actor_face_paths(actor_dir, source_video_path)
        image = filtered_faces[0] if source_video_path else _representative_image(actor_dir)
        gallery.append((str(image), actor_name))

    return gallery


def load_actor_library(
    sort_by="name", sort_order="asc", source_video_path=None, actor_name_filter=None
):
    gallery = load_actor_gallery(sort_by, sort_order, source_video_path, actor_name_filter)
    return gallery, f"出演者数：{len(gallery)}"


def get_actor_details(actor_name, source_video_path=None):
    actor_dir = _actor_dir(actor_name)

    if actor_dir is None:
        return "", [], "出演者を選択してください"

    faces = _actor_face_paths(actor_dir, source_video_path)
    metadata_path = actor_dir / "faces_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    gallery = [
        (
            str(face),
            (
                f"{face.name} | 元動画: {metadata.get(face.name, {}).get('video_name', '元動画情報なし')}"
                f" | 抽出位置: {metadata.get(face.name, {}).get('source_time_label', '情報なし')}"
            ),
        )
        for face in faces
    ]

    return actor_dir.name, gallery, f"顔画像：{len(gallery)}枚"


def _load_face_metadata(actor_dir):
    metadata_path = actor_dir / "faces_metadata.json"

    if not metadata_path.exists():
        return {}, metadata_path

    return json.loads(metadata_path.read_text(encoding="utf-8")), metadata_path


def _refresh_actor_after_face_change(actor_dir):
    faces_dir = actor_dir / "faces"
    faces = sorted(faces_dir.glob("*.jpg")) if faces_dir.exists() else []

    if not faces:
        shutil.rmtree(actor_dir)
        invalidate_person_index()
        return False

    shutil.copy2(faces[0], actor_dir / "representative.jpg")
    _update_actor_embedding(actor_dir.name)
    return True


def _validate_actor_face(actor_name, selected_face_path):
    actor_dir = _actor_dir(actor_name)

    if actor_dir is None or not selected_face_path:
        return None, None

    face_path = Path(selected_face_path)
    faces_dir = actor_dir / "faces"

    try:
        face_path.resolve().relative_to(faces_dir.resolve())
    except ValueError:
        return None, None

    if face_path.suffix.lower() != ".jpg" or not face_path.is_file():
        return None, None

    return actor_dir, face_path


def remove_actor_face(actor_name, selected_face_path):
    actor_dir, face_path = _validate_actor_face(actor_name, selected_face_path)

    if actor_dir is None:
        return "出演者ライブラリから除外する顔画像を選択してください"

    metadata, metadata_path = _load_face_metadata(actor_dir)
    embedding_path = face_path.with_suffix(".npy")
    face_path.unlink()

    if embedding_path.exists():
        embedding_path.unlink()

    metadata.pop(face_path.name, None)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    actor_exists = _refresh_actor_after_face_change(actor_dir)

    if actor_exists:
        return f"{face_path.name} を {actor_name} の出演者ライブラリから除外しました"

    return f"{face_path.name} を除外し、顔画像がなくなったため {actor_name} を削除しました"


def set_actor_representative_image(actor_name, selected_face_path):
    actor_dir, face_path = _validate_actor_face(actor_name, selected_face_path)

    if actor_dir is None:
        return "サムネイルに設定する顔画像を選択してください"

    shutil.copy2(face_path, actor_dir / "representative.jpg")
    return f"{actor_name} のサムネイルを {face_path.name} に変更しました"


def reclassify_actor_face(source_actor_name, selected_face_path, target_actor_name):
    source_dir, face_path = _validate_actor_face(source_actor_name, selected_face_path)

    if source_dir is None:
        return "再分類する顔画像を選択してください"

    target_actor_name, error = _validate_reclassification_target(target_actor_name)

    if error:
        return error

    if target_actor_name == source_actor_name:
        return "再分類先には別の出演者を選択してください"

    target_dir = PERSON_LIBRARY_DIR / target_actor_name
    target_dir.mkdir(parents=True, exist_ok=True)

    source_metadata, source_metadata_path = _load_face_metadata(source_dir)
    target_metadata, target_metadata_path = _load_face_metadata(target_dir)
    target_faces_dir = target_dir / "faces"
    target_faces_dir.mkdir(parents=True, exist_ok=True)

    destination = target_faces_dir / face_path.name
    index = 1

    while destination.exists():
        destination = target_faces_dir / f"{face_path.stem}_{index}{face_path.suffix}"
        index += 1

    embedding_path = face_path.with_suffix(".npy")
    shutil.move(str(face_path), str(destination))

    if embedding_path.exists():
        shutil.move(str(embedding_path), str(destination.with_suffix(".npy")))

    target_metadata[destination.name] = source_metadata.pop(
        face_path.name,
        {"video_name": "元動画情報なし", "video_path": ""},
    )
    source_metadata_path.write_text(
        json.dumps(source_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    target_metadata_path.write_text(
        json.dumps(target_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    source_exists = _refresh_actor_after_face_change(source_dir)
    _refresh_actor_after_face_change(target_dir)

    if source_exists:
        return f"{destination.name} を {source_actor_name} から {target_actor_name} へ再分類しました"

    return f"{destination.name} を {target_actor_name} へ再分類し、空になった {source_actor_name} を削除しました"


def _validate_reclassification_target(actor_name):
    actor_name = (actor_name or "").strip()

    if not actor_name:
        return None, "移動先の出演者名を入力してください"

    if actor_name in {".", ".."} or re.search(r'[<>:"/\\|?*]', actor_name):
        return None, "出演者名に使用できない文字が含まれています"

    return actor_name, None


def reclassify_actor_faces(source_actor_name, selected_face_paths, target_actor_name):
    source_dir = _actor_dir(source_actor_name)
    target_actor_name, error = _validate_reclassification_target(target_actor_name)

    if source_dir is None:
        return "移動元の出演者を選択してください"

    if error:
        return error

    if target_actor_name == source_actor_name:
        return "移動先には別の出演者名を指定してください"

    selected_paths = selected_face_paths or []
    face_paths = []
    seen = set()

    for selected_face_path in selected_paths:
        _, face_path = _validate_actor_face(source_actor_name, selected_face_path)

        if face_path is None or face_path in seen:
            continue

        seen.add(face_path)
        face_paths.append(face_path)

    if not face_paths:
        return "移動する顔画像を1枚以上選択してください"

    target_dir = PERSON_LIBRARY_DIR / target_actor_name
    target_faces_dir = target_dir / "faces"
    target_faces_dir.mkdir(parents=True, exist_ok=True)
    source_metadata, source_metadata_path = _load_face_metadata(source_dir)
    target_metadata, target_metadata_path = _load_face_metadata(target_dir)

    moved = 0

    for face_path in face_paths:
        destination = target_faces_dir / face_path.name
        index = 1

        while destination.exists():
            destination = target_faces_dir / f"{face_path.stem}_{index}{face_path.suffix}"
            index += 1

        embedding_path = face_path.with_suffix(".npy")
        shutil.move(str(face_path), str(destination))

        if embedding_path.exists():
            shutil.move(str(embedding_path), str(destination.with_suffix(".npy")))

        target_metadata[destination.name] = source_metadata.pop(
            face_path.name,
            {"video_name": "元動画情報なし", "video_path": ""},
        )
        moved += 1

    source_metadata_path.write_text(
        json.dumps(source_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    target_metadata_path.write_text(
        json.dumps(target_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_exists = _refresh_actor_after_face_change(source_dir)
    _refresh_actor_after_face_change(target_dir)

    if source_exists:
        return f"{moved}枚を {source_actor_name} から {target_actor_name} へ移動しました"

    return f"{moved}枚を {target_actor_name} へ移動し、空になった {source_actor_name} を削除しました"


def get_similar_actor_choices(actor_name):
    actor_dir = _actor_dir(actor_name)

    if actor_dir is None:
        return []

    source_embedding = actor_dir / "representative.npy"

    if not source_embedding.exists():
        return [name for name in load_actor_list() if name != actor_name]

    source = np.load(source_embedding)
    source_norm = np.linalg.norm(source)

    if source_norm == 0:
        return [name for name in load_actor_list() if name != actor_name]

    candidates = []

    for candidate_name in load_actor_list():
        if candidate_name == actor_name:
            continue

        embedding_path = PERSON_LIBRARY_DIR / candidate_name / "representative.npy"

        if not embedding_path.exists():
            continue

        candidate = np.load(embedding_path)
        denominator = source_norm * np.linalg.norm(candidate)

        if denominator == 0:
            continue

        score = float(np.dot(source, candidate) / denominator)
        candidates.append((score, candidate_name))

    candidates.sort(reverse=True)
    return [name for _, name in candidates]


def rename_actor(actor_name, new_name):
    actor_dir = _actor_dir(actor_name)
    new_name = (new_name or "").strip()

    if actor_dir is None:
        return actor_name, "出演者を選択してください"

    if not new_name:
        return actor_name, "名前を入力してください"

    if new_name in {".", ".."} or re.search(r'[<>:"/\\|?*]', new_name):
        return actor_name, "名前に使用できない文字が含まれています"

    destination = PERSON_LIBRARY_DIR / new_name

    if destination.exists() and destination != actor_dir:
        return actor_name, "同じ名前の出演者が既に存在します"

    if destination == actor_dir:
        return actor_name, "名前は変更されていません"

    actor_dir.rename(destination)
    return new_name, "名前を変更しました"


def delete_actor(actor_name):
    actor_dir = _actor_dir(actor_name)

    if actor_dir is None:
        return "出演者を選択してください"

    shutil.rmtree(actor_dir)
    return f"{actor_name} を削除しました"


def delete_actors(actor_names):
    selected_names = []
    seen = set()
    for actor_name in actor_names or []:
        if actor_name in seen:
            continue
        seen.add(actor_name)
        if _actor_dir(actor_name) is not None:
            selected_names.append(actor_name)

    if not selected_names:
        return 0, "削除する出演者を1人以上選択してください"

    for actor_name in selected_names:
        shutil.rmtree(PERSON_LIBRARY_DIR / actor_name)
    invalidate_person_index()
    return len(selected_names), f"{len(selected_names)}人の出演者を削除しました"


def merge_actors_into_target(source_names, target_name):
    target_dir = _actor_dir(target_name)
    if target_dir is None:
        return 0, "統合先の出演者を選択してください"

    source_dirs = []
    seen = set()
    for source_name in source_names or []:
        if source_name in seen or source_name == target_name:
            continue
        seen.add(source_name)
        source_dir = _actor_dir(source_name)
        if source_dir is not None:
            source_dirs.append(source_dir)
    if not source_dirs:
        return 0, "統合元の出演者を1人以上選択してください"

    related_video_names = _source_video_names(target_dir, *source_dirs)
    target_embedding_path = target_dir / "representative.npy"
    similarities = []
    moved = 0
    merged = 0

    for source_dir in source_dirs:
        source_embedding_path = source_dir / "representative.npy"
        if source_embedding_path.exists() and target_embedding_path.exists():
            similarity = _similarity(np.load(source_embedding_path), np.load(target_embedding_path))
            if similarity is not None:
                similarities.append(similarity)
        _, moved_faces = _merge_actor_files(source_dir, target_dir)
        moved += moved_faces
        merged += 1

    threshold = get_similarity_threshold()
    if similarities:
        threshold = calibrate_similarity_threshold(float(np.median(similarities)))
    related_moved, related_checked = _reclassify_related_generated_faces(
        target_dir, related_video_names, threshold
    )
    message = f"{merged}人を {target_name} に統合しました（顔画像：{moved}枚）"
    if similarities:
        message += f"。自動照合しきい値を {threshold:.2f} に調整しました"
    if related_checked:
        message += f"。同じ元動画を{related_checked}枚再照合し、{related_moved}枚を自動再分類しました"
    return merged, message


def merge_actors(source_name, target_name, result_name=None):
    source_dir = _actor_dir(source_name)
    target_dir = _actor_dir(target_name)
    result_name = (result_name or target_name or "").strip()

    if source_dir is None or target_dir is None:
        return "統合する出演者を選択してください"

    if source_dir == target_dir:
        return "統合先には別の出演者を選択してください"

    if result_name in {".", ".."} or re.search(r'[<>:"/\\|?*]', result_name):
        return "統合後の名称に使用できない文字が含まれています"

    result_dir = PERSON_LIBRARY_DIR / result_name

    if result_dir.exists() and result_dir not in {source_dir, target_dir}:
        return "統合後の名称と同じ出演者が既に存在します"

    source_embedding = source_dir / "representative.npy"
    target_embedding = target_dir / "representative.npy"
    adjusted_threshold = None
    related_video_names = _source_video_names(source_dir, target_dir)

    if source_embedding.exists() and target_embedding.exists():
        source = np.load(source_embedding)
        target = np.load(target_embedding)
        similarity = _similarity(source, target)

        if similarity is not None:
            adjusted_threshold = calibrate_similarity_threshold(similarity)

    _, moved = _merge_actor_files(source_dir, target_dir)

    if result_dir != target_dir:
        target_dir.rename(result_dir)
        target_name = result_name
        target_dir = result_dir

    message = f"{source_name} を {target_name} に統合しました（顔画像：{moved}枚）"

    if adjusted_threshold is not None:
        message += f"。自動照合しきい値を {adjusted_threshold:.2f} に調整しました"
        # Reclassifying every actor after each manual merge can cascade through
        # a large library and make this operation appear to never finish.
        # The new threshold is persisted immediately and applies to subsequent
        # automatic matching; existing actors remain unchanged unless the user
        # explicitly chooses a bulk reclassification operation.
        message += "。以降の自動照合には新しいしきい値を使用します"

    related_moved, related_checked = _reclassify_related_generated_faces(
        target_dir,
        related_video_names,
        adjusted_threshold if adjusted_threshold is not None else get_similarity_threshold(),
    )
    if related_checked:
        message += (
            f"。同じ元動画の未命名出演者を{related_checked}枚再照合し、"
            f"{related_moved}枚を自動再分類しました"
        )

    return message
