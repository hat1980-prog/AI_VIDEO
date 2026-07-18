import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.cluster import DBSCAN

from config import CLUSTER_EPS, EMBEDDINGS_DIR, FACES_DIR
from services.video_service import get_video_id


def cluster_faces(video_name):
    if not video_name:
        return "動画を選択してください"

    video_id = get_video_id(video_name)

    if video_id is None:
        return "動画ファイルが見つかりません"

    embedding_dir = EMBEDDINGS_DIR / video_id
    face_dir = FACES_DIR / video_id

    if not embedding_dir.exists():
        return "Embeddingフォルダがありません"

    embedding_files = sorted(embedding_dir.glob("*.npy"))

    if not embedding_files:
        return "Embeddingがありません"

    assignments_path = face_dir / "assignments.json"

    if assignments_path.exists():
        assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    else:
        assignments = {}

    embeddings = []
    image_names = []

    for embedding_file in embedding_files:
        image_name = f"{embedding_file.stem}.jpg"

        if image_name in assignments:
            continue

        embeddings.append(np.load(embedding_file))
        image_names.append(image_name)

    labels = []

    if embeddings:
        labels = DBSCAN(
            eps=CLUSTER_EPS,
            min_samples=2,
            metric="cosine"
        ).fit_predict(embeddings)

    for pattern in ("Actor_*", "Person_*"):
        for folder in face_dir.glob(pattern):
            shutil.rmtree(folder)

    unknown_dir = face_dir / "Unknown"

    if unknown_dir.exists():
        shutil.rmtree(unknown_dir)

    created = set(assignments.values())

    for actor_name in created:
        (face_dir / f"Actor_{actor_name}").mkdir(exist_ok=True)

    for label in labels:
        if label == -1:
            (face_dir / "Unknown").mkdir(exist_ok=True)
        else:
            (face_dir / f"Person_{label}").mkdir(exist_ok=True)
            created.add(f"Person_{label}")

    result = []

    for image_name, actor_name in assignments.items():
        source = face_dir / image_name

        if not source.exists():
            continue

        shutil.copy2(source, face_dir / f"Actor_{actor_name}" / image_name)
        result.append(f"{image_name} → {actor_name}")

    for image_name, label in zip(image_names, labels):
        source = face_dir / image_name

        if not source.exists():
            continue

        if label == -1:
            destination = face_dir / "Unknown" / image_name
            person = "Unknown"
        else:
            destination = face_dir / f"Person_{label}" / image_name
            person = f"Person_{label}"

        shutil.copy2(source, destination)
        result.append(f"{image_name} → {person}")

    summary = [
        "=== Cluster Result ===",
        "",
        f"顔画像 : {len(embedding_files)}",
        f"出演者ライブラリ一致 : {len(assignments)}",
        f"人物数 : {len(created)}",
        "",
    ]
    summary.extend(result)

    return "\n".join(summary)
