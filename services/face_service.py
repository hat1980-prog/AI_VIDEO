from pathlib import Path
import cv2
import numpy as np
import shutil

from insightface.app import FaceAnalysis

from config import (
    FRAMES_DIR,
    FACES_DIR,
    EMBEDDINGS_DIR,
    FACE_MODEL,
    FACE_CTX
)

from config import PERSON_LIBRARY_DIR

from services.person_library_service import (
    create_person,
    find_similar_actor,
    update_actor_embedding
)

# InsightFace初期化
app = FaceAnalysis(name=FACE_MODEL)
app.prepare(ctx_id=FACE_CTX)


def extract_faces(video_name):

    frame_dir = FRAMES_DIR / Path(video_name).stem
    face_dir = FACES_DIR / Path(video_name).stem
    embedding_dir = EMBEDDINGS_DIR / Path(video_name).stem

    face_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    frame_files = sorted(frame_dir.glob("*.jpg"))

    if len(frame_files) == 0:
        return "フレーム画像がありません"

    for image_file in frame_files:

        img = cv2.imread(str(image_file))

        if img is None:
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

    update_actor_embedding(actor_name)

    return f"{count}枚の顔画像とEmbeddingを保存しました"