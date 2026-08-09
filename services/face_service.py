from pathlib import Path
import json
import cv2
import numpy as np
import shutil
import time

from insightface.app import FaceAnalysis
from insightface.utils import face_align


def _estimate_norm_with_current_api(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    destination = face_align.arcface_dst * ratio
    destination[:, 0] += diff_x
    transform = face_align.trans.SimilarityTransform.from_estimate(lmk, destination)
    return transform.params[0:2, :]


if hasattr(face_align.trans.SimilarityTransform, "from_estimate"):
    face_align.estimate_norm = _estimate_norm_with_current_api

from config import (
    FRAMES_DIR,
    FACES_DIR,
    EMBEDDINGS_DIR,
    FACE_MODEL,
    FACE_CTX,
    MIN_FACE_SIZE,
    FRAME_INTERVAL,
)

from config import PERSON_LIBRARY_DIR

from services.person_library_service import (
    create_person,
    find_similar_actor,
    update_actor_embedding
)
from services.video_service import get_video_id, get_video_path
from services.exclusion_service import is_excluded_face
from services.source_video_service import normalize_source_video_paths

# InsightFace初期化
app = FaceAnalysis(name=FACE_MODEL)
app.prepare(ctx_id=FACE_CTX)


def remove_small_faces(face_dir, embedding_dir):
    removed = []

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

        removed.append((face_path.relative_to(face_dir), width, height))

    return removed


def _format_source_time(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"約{hours}時間{minutes}分{seconds:02d}秒"

    return f"約{minutes}分{seconds:02d}秒"


def _get_frame_time_info(image_file, frame_interval):
    try:
        frame_index = int(image_file.stem)
    except ValueError:
        frame_index = 1

    seconds = max(0, frame_index - 1) * frame_interval
    return {
        "source_time_seconds": seconds,
        "source_time_label": _format_source_time(seconds),
        "source_time_basis": "sample_interval",
    }


def extract_faces(video_name, progress_callback=None, should_cancel=None, frame_interval=None):
    started_at = time.perf_counter()

    video_id = get_video_id(video_name)

    if video_id is None:
        return "動画ファイルが見つかりません"

    video_path = get_video_path(video_name)

    frame_dir = FRAMES_DIR / video_id
    face_dir = FACES_DIR / video_id
    embedding_dir = EMBEDDINGS_DIR / video_id

    face_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    removed_existing_faces = remove_small_faces(face_dir, embedding_dir)

    count = 0
    skipped_small_faces = []
    skipped_excluded_faces = 0
    matched_actors = {}
    updated_actors = set()
    face_metadata = {}
    actor_metadata = {}

    frame_files = sorted(frame_dir.glob("*.jpg"))

    if len(frame_files) == 0:
        return "フレーム画像がありません"

    total_frames = len(frame_files)
    frame_metadata_path = frame_dir / "frames_metadata.json"

    if frame_metadata_path.exists():
        frame_metadata = json.loads(frame_metadata_path.read_text(encoding="utf-8"))
        saved_interval = frame_metadata.get("frame_interval_seconds")
    else:
        saved_interval = None

    try:
        extraction_interval = int(saved_interval or frame_interval or FRAME_INTERVAL)
    except (TypeError, ValueError):
        extraction_interval = FRAME_INTERVAL

    processed_frames = 0
    cancelled = False

    for frame_index, image_file in enumerate(frame_files, start=1):

        if should_cancel and should_cancel():
            cancelled = True
            break

        processed_frames += 1

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
                skipped_small_faces.append((f"{image_file.name} #{i + 1}", crop_width, crop_height))
                continue

            if is_excluded_face(face.embedding):
                skipped_excluded_faces += 1
                continue

            face_filename = f"{image_file.stem}_{i+1}.jpg"
            face_metadata[face_filename] = {
                "video_name": video_path.name,
                "video_path": str(video_path),
                **_get_frame_time_info(image_file, extraction_interval),
            }

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

            if actor_name not in actor_metadata:
                metadata_path = PERSON_LIBRARY_DIR / actor_name / "faces_metadata.json"

                if metadata_path.exists():
                    actor_metadata[actor_name] = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                else:
                    actor_metadata[actor_name] = {}

            actor_face_path = actor_faces / face_filename
            index = 1

            while actor_face_path.exists():
                actor_face_path = actor_faces / f"{Path(face_filename).stem}_{index}.jpg"
                index += 1

            shutil.copy2(
                face_path,
                actor_face_path
            )

            shutil.copy2(
                embedding_path,
                actor_face_path.with_suffix(".npy")
            )
            actor_metadata[actor_name][actor_face_path.name] = face_metadata[face_filename]

        if progress_callback:
            progress_callback(frame_index / total_frames)

    for actor_name in updated_actors:
        update_actor_embedding(actor_name)

    (face_dir / "assignments.json").write_text(
        json.dumps(matched_actors, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (face_dir / "faces_metadata.json").write_text(
        json.dumps(face_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    for actor_name, metadata in actor_metadata.items():
        (PERSON_LIBRARY_DIR / actor_name / "faces_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    normalized_actor_faces, normalized_extracted_faces = normalize_source_video_paths(video_path)

    low_resolution_faces = [
        *(f"新規: {name} ({width}x{height}px)" for name, width, height in skipped_small_faces),
        *(f"既存: {name} ({width}x{height}px)" for name, width, height in removed_existing_faces),
    ]
    log_limit = 50
    low_resolution_log = ""

    if low_resolution_faces:
        displayed = low_resolution_faces[:log_limit]
        low_resolution_log = "\n低解像度のため除外した顔:\n" + "\n".join(displayed)

        if len(low_resolution_faces) > log_limit:
            low_resolution_log += f"\n…ほか{len(low_resolution_faces) - log_limit}件"

    message = (
        f"{count}枚の顔画像とEmbeddingを保存しました"
        f"（小さすぎる顔を{len(skipped_small_faces)}枚除外、"
        f"除外済み人物を{skipped_excluded_faces}枚無視、"
        f"既存の低解像度顔を{len(removed_existing_faces)}枚削除、"
        f"出演者ライブラリ一致を{len(matched_actors)}枚）"
        f"\n処理フレーム: {processed_frames}/{total_frames}、所要時間: {time.perf_counter() - started_at:.1f}秒"
        f"\n同名動画の元動画パス統合: 出演者ライブラリ{normalized_actor_faces}枚、"
        f"抽出顔{normalized_extracted_faces}枚"
        f"{low_resolution_log}"
    )

    if cancelled:
        message += "\n中断要求を受けたため、ここまでの顔抽出結果を保存して終了しました"

    return message
