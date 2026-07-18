from pathlib import Path
import json
import cv2
import numpy as np
import shutil

from insightface.app import FaceAnalysis

from config import (
    FRAMES_DIR,
    FACES_DIR,
    EMBEDDINGS_DIR,
    FACE_MODEL,
    FACE_CTX,
    MIN_FACE_SIZE,
)

from config import PERSON_LIBRARY_DIR

from services.person_library_service import (
    create_person,
    find_similar_actor,
    update_actor_embedding
)
from services.video_service import get_video_id
from services.exclusion_service import is_excluded_face

# InsightFace初期化
app = FaceAnalysis(name=FACE_MODEL)
app.prepare(ctx_id=FACE_CTX)


def remove_small_faces(face_dir, embedding_dir):
    removed = 0

    for face_path in face_dir.rglob("*.jpg"):
        image = cv2.imread(str(face_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        if min(height, width) >= MIN_FACE_SIZE:
            continue

        face_path.unlink()

        if face_path.parent == face_dir:
            embedding_path = embedding_dir / f"{face_path.stem}.npy"

            if embedding_path.exists():
                embedding_path.unlink()

        removed += 1

    return removed


def extract_faces(video_name, progress_callback=None):

    video_id = get_video_id(video_name)

    if video_id is None:
        return "動画ファイルが見つかりません"

    frame_dir = FRAMES_DIR / video_id
    face_dir = FACES_DIR / video_id
    embedding_dir = EMBEDDINGS_DIR / video_id

    face_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    removed_existing_faces = remove_small_faces(face_dir, embedding_dir)

    count = 0
    skipped_small_faces = 0
    skipped_excluded_faces = 0
    matched_actors = {}
    updated_actors = set()

    frame_files = sorted(frame_dir.glob("*.jpg"))

    if len(frame_files) == 0:
        return "フレーム画像がありません"

    total_frames = len(frame_files)

    for frame_index, image_file in enumerate(frame_files, start=1):

        img = cv2.imread(str(image_file))

        if img is None:
            if progress_callback:
                progress_callback(frame_index / total_frames)
            continue

        detected_faces = app.get(img)

        for i, face in enumerate(detected_faces):

            x1, y1, x2, y2 = map(int, face.bbox)

            h, w = img.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_height, crop_width = crop.shape[:2]

            if min(crop_height, crop_width) < MIN_FACE_SIZE:
                skipped_small_faces += 1
                continue

            if is_excluded_face(face.embedding):
                skipped_excluded_faces += 1
                continue

            face_filename = f"{image_file.stem}_{i+1}.jpg"

            face_path = face_dir / face_filename

            cv2.imwrite(str(face_path), crop)

            embedding_path = embedding_dir / f"{image_file.stem}_{i+1}.npy"

            np.save(
                embedding_path,
                face.embedding
            )

            count += 1

            actor = find_similar_actor(face.embedding)

            if actor is None:

                actor_name = create_person(
                    face_path,
                    embedding_path
                )

            else:

                actor_name = actor["name"]
                matched_actors[face_filename] = actor_name

            updated_actors.add(actor_name)

            actor_faces = (
                PERSON_LIBRARY_DIR /
                actor_name /
                "faces"
            )

            actor_faces.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                face_path,
                actor_faces / face_filename
            )

            shutil.copy2(
                embedding_path,
                actor_faces / Path(face_filename).with_suffix(".npy")
            )

        if progress_callback:
            progress_callback(frame_index / total_frames)

    for actor_name in updated_actors:
        update_actor_embedding(actor_name)

    (face_dir / "assignments.json").write_text(
        json.dumps(matched_actors, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return (
        f"{count}枚の顔画像とEmbeddingを保存しました"
        f"（小さすぎる顔を{skipped_small_faces}枚除外、"
        f"除外済み人物を{skipped_excluded_faces}枚無視、"
        f"既存の低解像度顔を{removed_existing_faces}枚削除）"
    )
