import json
import shutil

import numpy as np

from config import (
    PERSON_LIBRARY_DIR,
    PERSON_LIBRARY_SETTINGS_PATH,
    PERSON_SIMILARITY_MARGIN,
    PERSON_SIMILARITY_THRESHOLD,
    PERSON_SIMILARITY_THRESHOLD_MAX,
    PERSON_SIMILARITY_THRESHOLD_MIN,
)


_PERSON_INDEX = {"fingerprint": None, "names": [], "images": [], "embeddings": None}


def _invalidate_person_index():
    _PERSON_INDEX["fingerprint"] = None


def invalidate_person_index():
    _invalidate_person_index()


def _get_person_index():
    records = []

    for folder in sorted(PERSON_LIBRARY_DIR.iterdir()):
        if not folder.is_dir():
            continue

        image = folder / "representative.jpg"
        embedding = folder / "representative.npy"

        if image.exists() and embedding.exists():
            stat = embedding.stat()
            records.append((folder.name, image, embedding, stat.st_mtime_ns, stat.st_size))

    fingerprint = tuple((name, modified, size) for name, _, _, modified, size in records)

    if fingerprint == _PERSON_INDEX["fingerprint"]:
        return _PERSON_INDEX

    names = []
    images = []
    embeddings = []

    for name, image, embedding_path, _, _ in records:
        embedding = np.load(embedding_path)
        norm = np.linalg.norm(embedding)

        if norm == 0:
            continue

        names.append(name)
        images.append(image)
        embeddings.append(embedding / norm)

    _PERSON_INDEX.update(
        fingerprint=fingerprint,
        names=names,
        images=images,
        embeddings=np.stack(embeddings) if embeddings else None,
    )
    return _PERSON_INDEX


def get_next_person_id():
    numbers = []

    for folder in PERSON_LIBRARY_DIR.glob("Person*"):
        try:
            numbers.append(int(folder.name.replace("Person", "")))
        except ValueError:
            continue

    return max(numbers, default=0) + 1


def create_person(face_path, embedding_path):
    person_name = f"Person{get_next_person_id():06d}"
    person_dir = PERSON_LIBRARY_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(face_path, person_dir / "representative.jpg")
    shutil.copy2(embedding_path, person_dir / "representative.npy")
    _invalidate_person_index()
    return person_name


def load_all_persons():
    index = _get_person_index()

    if index["embeddings"] is None:
        return []

    return [
        {"name": name, "image": image, "embedding": embedding}
        for name, image, embedding in zip(index["names"], index["images"], index["embeddings"])
    ]


def get_similarity_threshold():
    if not PERSON_LIBRARY_SETTINGS_PATH.exists():
        return PERSON_SIMILARITY_THRESHOLD

    try:
        return float(json.loads(PERSON_LIBRARY_SETTINGS_PATH.read_text(encoding="utf-8"))["similarity_threshold"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return PERSON_SIMILARITY_THRESHOLD


def calibrate_similarity_threshold(similarity):
    # A confirmed manual merge is a fresh positive example.  Do not constrain
    # the result by the previous value: doing so made the threshold monotonic
    # decreasing and permanently pinned it at its minimum (0.45).
    adjusted_threshold = max(
        PERSON_SIMILARITY_THRESHOLD_MIN,
        min(PERSON_SIMILARITY_THRESHOLD_MAX, similarity - PERSON_SIMILARITY_MARGIN),
    )
    PERSON_LIBRARY_SETTINGS_PATH.write_text(
        json.dumps({"similarity_threshold": adjusted_threshold}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return adjusted_threshold


def find_similar_actor(embedding, threshold=None):
    threshold = get_similarity_threshold() if threshold is None else threshold
    index = _get_person_index()

    if index["embeddings"] is None:
        return None

    embedding = np.asarray(embedding)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    scores = index["embeddings"] @ (embedding / norm)
    best_index = int(np.argmax(scores))

    if float(scores[best_index]) < threshold:
        return None

    return {
        "name": index["names"][best_index],
        "image": index["images"][best_index],
        "embedding": index["embeddings"][best_index],
        "similarity": float(scores[best_index]),
    }


def update_actor_embedding(actor_name):
    actor_dir = PERSON_LIBRARY_DIR / actor_name
    embeddings = []

    for embedding_path in sorted((actor_dir / "faces").glob("*.npy")):
        embeddings.append(np.load(embedding_path))

    if not embeddings:
        return False

    mean_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_embedding)

    if norm == 0:
        return False

    np.save(actor_dir / "representative.npy", mean_embedding / norm)
    _invalidate_person_index()
    return True
