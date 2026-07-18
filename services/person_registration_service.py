from pathlib import Path
import re
import shutil

import numpy as np

from config import EMBEDDINGS_DIR, FACES_DIR, PERSON_LIBRARY_DIR
from services.person_library_service import update_actor_embedding
from services.video_service import get_video_id


def _validate_actor_name(actor_name):
    actor_name = (actor_name or "").strip()

    if not actor_name:
        return None, "出演者名を入力してください"

    if actor_name in {".", ".."} or re.search(r'[<>:"/\\|?*]', actor_name):
        return None, "出演者名に使用できない文字が含まれています"

    return actor_name, None


def _remove_duplicate_faces(embeddings, target_dir):
    affected_actors = set()

    for actor_dir in PERSON_LIBRARY_DIR.iterdir():
        if not actor_dir.is_dir() or actor_dir == target_dir:
            continue

        faces_dir = actor_dir / "faces"

        if not faces_dir.exists():
            continue

        for embedding_path in list(faces_dir.glob("*.npy")):
            stored_embedding = np.load(embedding_path)

            if not any(np.array_equal(stored_embedding, embedding) for embedding in embeddings):
                continue

            image_path = embedding_path.with_suffix(".jpg")
            embedding_path.unlink()

            if image_path.exists():
                image_path.unlink()

            affected_actors.add(actor_dir.name)

    for actor_name in affected_actors:
        actor_dir = PERSON_LIBRARY_DIR / actor_name
        faces_dir = actor_dir / "faces"

        if any(faces_dir.glob("*.jpg")):
            update_actor_embedding(actor_name)
        else:
            shutil.rmtree(actor_dir)


def register_cluster_as_actor(video, cluster_name, actor_name):
    video_id = get_video_id(video)
    actor_name, error = _validate_actor_name(actor_name)

    if video_id is None:
        return "動画ファイルが見つかりません"

    if error:
        return error

    if not cluster_name or not re.fullmatch(r"Person_\d+", cluster_name):
        return "人物一覧からPersonを選択してください"

    source_dir = FACES_DIR / video_id / cluster_name
    embedding_dir = EMBEDDINGS_DIR / video_id

    if not source_dir.is_dir():
        return "選択した人物クラスタが見つかりません"

    records = []

    for image_path in sorted(source_dir.glob("*.jpg")):
        embedding_path = embedding_dir / f"{image_path.stem}.npy"

        if embedding_path.exists():
            records.append((image_path, embedding_path, np.load(embedding_path)))

    if not records:
        return "登録できる顔画像とEmbeddingがありません"

    target_dir = PERSON_LIBRARY_DIR / actor_name
    target_faces_dir = target_dir / "faces"
    target_faces_dir.mkdir(parents=True, exist_ok=True)

    _remove_duplicate_faces([embedding for _, _, embedding in records], target_dir)

    added = 0

    for image_path, embedding_path, _ in records:
        destination = target_faces_dir / image_path.name
        index = 1

        while destination.exists():
            destination = target_faces_dir / f"{image_path.stem}_{index}{image_path.suffix}"
            index += 1

        shutil.copy2(image_path, destination)
        shutil.copy2(embedding_path, destination.with_suffix(".npy"))
        added += 1

    representative_image = target_dir / "representative.jpg"

    if not representative_image.exists():
        shutil.copy2(records[0][0], representative_image)

    update_actor_embedding(actor_name)

    return f"{cluster_name} を {actor_name} として登録しました（顔画像：{added}枚）"
