from pathlib import Path
import re
import shutil

import numpy as np

from config import PERSON_LIBRARY_DIR
from services.person_library_service import calibrate_similarity_threshold, invalidate_person_index


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

            moved += 1

    source_name = source_dir.name
    target_name = target_dir.name
    shutil.rmtree(source_dir)
    _update_actor_embedding(target_name)

    return source_name, moved


def reclassify_actors(threshold, progress_callback=None):
    merged = []

    while True:
        actor_names = load_actor_list()
        merged_pair = None

        for target_name in actor_names:
            target_dir = _actor_dir(target_name)
            target_embedding_path = target_dir / "representative.npy"

            if not target_embedding_path.exists():
                continue

            target_embedding = np.load(target_embedding_path)

            for source_name in actor_names:
                if source_name == target_name:
                    continue

                if (
                    not _is_generated_actor_name(source_name)
                    and not _is_generated_actor_name(target_name)
                ):
                    continue

                source_dir = _actor_dir(source_name)
                source_embedding_path = source_dir / "representative.npy"

                if not source_embedding_path.exists():
                    continue

                source_embedding = np.load(source_embedding_path)
                similarity = _similarity(source_embedding, target_embedding)

                if similarity is not None and similarity >= threshold:
                    if (
                        _is_generated_actor_name(target_name)
                        and not _is_generated_actor_name(source_name)
                    ):
                        merged_pair = (target_dir, source_dir)
                    else:
                        merged_pair = (source_dir, target_dir)
                    break

            if merged_pair:
                break

        if not merged_pair:
            break

        source_name, _ = _merge_actor_files(*merged_pair)
        merged.append(source_name)

        if progress_callback:
            progress_callback(len(merged))

    return merged


def _actor_records():
    records = []

    for actor_dir in PERSON_LIBRARY_DIR.iterdir():
        if not actor_dir.is_dir() or not _representative_image(actor_dir):
            continue

        faces_dir = actor_dir / "faces"
        face_count = len(list(faces_dir.glob("*.jpg"))) if faces_dir.exists() else 0
        records.append((actor_dir.name, face_count))

    return records


def load_actor_list(sort_by="name", sort_order="asc"):
    records = _actor_records()
    reverse = sort_order == "desc"

    if sort_by == "count":
        records.sort(key=lambda record: (record[1], record[0].casefold()), reverse=reverse)
    else:
        records.sort(key=lambda record: record[0].casefold(), reverse=reverse)

    return [name for name, _ in records]


def load_actor_gallery(sort_by="name", sort_order="asc"):
    gallery = []

    for actor_name in load_actor_list(sort_by, sort_order):
        image = _representative_image(PERSON_LIBRARY_DIR / actor_name)
        gallery.append((str(image), actor_name))

    return gallery


def load_actor_library(sort_by="name", sort_order="asc"):
    gallery = load_actor_gallery(sort_by, sort_order)
    return gallery, f"出演者数：{len(gallery)}"


def get_actor_details(actor_name):
    actor_dir = _actor_dir(actor_name)

    if actor_dir is None:
        return "", [], "出演者を選択してください"

    faces_dir = actor_dir / "faces"
    faces = sorted(faces_dir.glob("*.jpg")) if faces_dir.exists() else []
    gallery = [(str(face), face.name) for face in faces]

    return actor_dir.name, gallery, f"顔画像：{len(gallery)}枚"


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

    message = f"{source_name} を {target_name} に統合しました（顔画像：{moved}枚）"

    if adjusted_threshold is not None:
        message += f"。自動照合しきい値を {adjusted_threshold:.2f} に調整しました"
        reclassified = reclassify_actors(adjusted_threshold)

        if reclassified:
            message += f"。再分類で{len(reclassified)}人を統合しました"

    return message
