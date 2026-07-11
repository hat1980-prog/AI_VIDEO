from pathlib import Path
import shutil

import numpy as np
from sklearn.cluster import DBSCAN

from config import EMBEDDINGS_DIR
from config import FACES_DIR


def cluster_faces(video_name):

    if not video_name:
        return "動画を選択してください"

    video_name = Path(video_name).stem

    embedding_dir = EMBEDDINGS_DIR / video_name
    face_dir = FACES_DIR / video_name

    if not embedding_dir.exists():
        return "Embeddingフォルダがありません"

    embedding_files = sorted(embedding_dir.glob("*.npy"))

    if len(embedding_files) == 0:
        return "Embeddingがありません"

    embeddings = []
    image_names = []

    for emb_file in embedding_files:

        embeddings.append(np.load(emb_file))

        image_names.append(emb_file.stem + ".jpg")

    labels = DBSCAN(
        eps=0.55,
        min_samples=2,
        metric="cosine"
    ).fit_predict(embeddings)

    # ---------- 古いPersonフォルダ削除 ----------
    for folder in face_dir.glob("Person_*"):
        shutil.rmtree(folder)

    unknown_dir = face_dir / "Unknown"

    if unknown_dir.exists():
        shutil.rmtree(unknown_dir)

    # ---------- フォルダ作成 ----------
    created = set()

    for label in labels:

        if label == -1:

            (face_dir / "Unknown").mkdir(
                exist_ok=True
            )

        else:

            folder = face_dir / f"Person_{label}"

            folder.mkdir(
                exist_ok=True
            )

            created.add(label)

    # ---------- 顔画像コピー ----------
    result = []

    for image_name, label in zip(image_names, labels):

        src = face_dir / image_name

        if not src.exists():
            continue

        if label == -1:

            dst = face_dir / "Unknown" / image_name

            person = "Unknown"

        else:

            dst = face_dir / f"Person_{label}" / image_name

            person = f"Person_{label}"

        shutil.copy2(src, dst)

        result.append(
            f"{image_name} → {person}"
        )

    summary = [
        "=== Cluster Result ===",
        "",
        f"顔画像 : {len(image_names)}",
        f"人物数 : {len(created)}",
        ""
    ]

    summary.extend(result)

    return "\n".join(summary)