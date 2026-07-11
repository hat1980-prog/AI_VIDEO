import os
import cv2
import json
import time
import sqlite3
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import whisper

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(r"B:\AI_VIDEO")
VIDEOS_DIR = BASE_DIR / "videos"
FRAMES_DIR = BASE_DIR / "frames"
FACES_DIR = BASE_DIR / "faces"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
TAGS_DIR = BASE_DIR / "tags"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "metadata.db"

FRAME_FPS = 0.5
MIN_FACE_SIZE = 80
CLUSTER_EPS = 0.45
CLUSTER_MIN_SAMPLES = 3

SUPPORTED_VIDEO_EXTENSIONS = [
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
]

# =========================================================
# DIRECTORY CREATE
# =========================================================

for directory in [
    VIDEOS_DIR,
    FRAMES_DIR,
    FACES_DIR,
    EMBEDDINGS_DIR,
    TRANSCRIPTS_DIR,
    TAGS_DIR,
    DATABASE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# =========================================================
# DATABASE
# =========================================================

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        path TEXT,
        duration REAL,
        transcript_path TEXT,
        tags_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""
)

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER,
        frame_path TEXT,
        face_path TEXT,
        cluster_id INTEGER,
        embedding_path TEXT,
        FOREIGN KEY(video_id) REFERENCES videos(id)
    )
"""
)

conn.commit()

# =========================================================
# LOAD MODELS
# =========================================================

print("Loading InsightFace...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

print("Loading Whisper...")
whisper_model = whisper.load_model("small")

print("Loading Qwen2-VL...")
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

qwen_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct"
)

# =========================================================
# HELPERS
# =========================================================


def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")


# =========================================================
# VIDEO INFO
# =========================================================


def get_video_duration(video_path):
    command = (
        f'ffprobe -v error -show_entries format=duration '
        f'-of default=noprint_wrappers=1:nokey=1 "{video_path}"'
    )

    result = subprocess.check_output(command, shell=True)
    return float(result.decode().strip())


# =========================================================
# FRAME EXTRACTION
# =========================================================


def extract_frames(video_path):
    video_name = Path(video_path).stem
    output_dir = FRAMES_DIR / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = list(output_dir.glob("*.jpg"))

    if len(existing_frames) > 0:
        print(f"Frames already exist: {video_name}")
        return output_dir

    output_pattern = output_dir / "%06d.jpg"

    command = (
        f'ffmpeg -hide_banner -loglevel error '
        f'-i "{video_path}" '
        f'-vf fps={FRAME_FPS} '
        f'"{output_pattern}"'
    )

    print(f"Extracting frames: {video_name}")
    run_command(command)

    return output_dir


# =========================================================
# FACE EXTRACTION
# =========================================================


def extract_faces(video_id, frames_dir):
    video_name = Path(frames_dir).name

    video_face_dir = FACES_DIR / video_name
    video_face_dir.mkdir(parents=True, exist_ok=True)

    embeddings = []
    metadata = []

    frame_files = sorted(list(Path(frames_dir).glob("*.jpg")))

    for frame_file in frame_files:
        img = cv2.imread(str(frame_file))

        if img is None:
            continue

        faces = face_app.get(img)

        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = bbox

            img_h, img_w = img.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            width = x2 - x1
            height = y2 - y1

            if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                continue

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            face_filename = f"{frame_file.stem}_{idx}.jpg"
            face_path = video_face_dir / face_filename

            cv2.imwrite(str(face_path), crop)

            embedding = face.embedding

            embedding_path = (
                EMBEDDINGS_DIR
                / f"{video_name}_{frame_file.stem}_{idx}.npy"
            )

            np.save(embedding_path, embedding)

            embeddings.append(embedding)
            metadata.append(
                {
                    "video_id": video_id,
                    "frame_path": str(frame_file),
                    "face_path": str(face_path),
                    "embedding_path": str(embedding_path),
                }
            )

    return embeddings, metadata


# =========================================================
# CLUSTERING
# =========================================================


def cluster_faces(embeddings, metadata):
    if len(embeddings) == 0:
        return

    embeddings_np = np.array(embeddings)

    clustering = DBSCAN(
        eps=CLUSTER_EPS,
        min_samples=CLUSTER_MIN_SAMPLES,
        metric="cosine",
    )

    labels = clustering.fit_predict(embeddings_np)

    for label, meta in zip(labels, metadata):
        try:
            cursor.execute(
                """
                INSERT INTO faces (
                    video_id,
                    frame_path,
                    face_path,
                    cluster_id,
                    embedding_path
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    meta["video_id"],
                    str(meta["frame_path"]),
                    str(meta["face_path"]),
                    int(label),
                    str(meta["embedding_path"]),
                ),
            )

            print(
                f"INSERT FACE: "
                f"cluster={label} "
                f"face={meta['face_path']}"
            )

        except Exception as e:
            import traceback
            print("FACE INSERT ERROR")
            print(meta)
            traceback.print_exc()

    conn.commit()

    print("FACES DB COMMIT COMPLETE")


# =========================================================
# WHISPER
# =========================================================


def transcribe_video(video_path):
    video_name = Path(video_path).stem

    transcript_path = TRANSCRIPTS_DIR / f"{video_name}.txt"

    if transcript_path.exists():
        print(f"Transcript already exists: {video_name}")
        return transcript_path

    print(f"Transcribing: {video_name}")

    result = whisper_model.transcribe(
        str(video_path),
        language="ja",
        fp16=True,
    )

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    return transcript_path


# =========================================================
# TAG GENERATION
# =========================================================


def generate_tags(frames_dir):
    frame_files = sorted(list(Path(frames_dir).glob("*.jpg")))

    if len(frame_files) == 0:
        return []

    selected = frame_files[::max(1, len(frame_files) // 8)]

    tags = set()

    for frame_file in selected[:4]:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(frame_file),
                        },
                        {
                            "type": "text",
                            "text": (
                                "この画像を解析し、日本語タグを10個以内で列挙してください。"
                                "人物、場所、状況、服装、人数、小物、背景などを簡潔に。"
                                "カンマ区切りで出力してください。"
                            ),
                        },
                    ],
                }
            ]

            text = qwen_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = qwen_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")

            generated_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=128,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = qwen_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            for tag in output_text.split(","):
                tag = tag.strip()

                if len(tag) > 0:
                    tags.add(tag)

        except Exception as e:
            print(f"Qwen tag error: {e}")

    torch.cuda.empty_cache()
    return sorted(list(tags))


# =========================================================
# PROCESS VIDEO
# =========================================================


def process_video(video_path):
    print("=" * 60)
    print(f"PROCESSING: {video_path}")
    print("=" * 60)

    filename = Path(video_path).name

    cursor.execute(
        "SELECT id FROM videos WHERE filename = ?",
        (filename,),
    )

    existing = cursor.fetchone()

    if existing:
        print(f"Already processed: {filename}")
        return

    duration = get_video_duration(video_path)

    cursor.execute(
        """
        INSERT INTO videos (
            filename,
            path,
            duration
        )
        VALUES (?, ?, ?)
    """,
        (
            filename,
            str(video_path),
            duration,
        ),
    )

    conn.commit()

    video_id = cursor.lastrowid

    # -----------------------------------------
    # FRAME EXTRACTION
    # -----------------------------------------

    frames_dir = extract_frames(video_path)

    # -----------------------------------------
    # FACE EXTRACTION
    # -----------------------------------------

    embeddings, metadata = extract_faces(video_id, frames_dir)

    # -----------------------------------------
    # CLUSTERING
    # -----------------------------------------

    cluster_faces(embeddings, metadata)

    # -----------------------------------------
    # TRANSCRIPTION
    # -----------------------------------------

    transcript_path = transcribe_video(video_path)

    # -----------------------------------------
    # TAG GENERATION
    # -----------------------------------------

    tags = generate_tags(frames_dir)

    tag_path = TAGS_DIR / f"{Path(video_path).stem}.json"

    with open(tag_path, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)

    # -----------------------------------------
    # UPDATE DB
    # -----------------------------------------

    cursor.execute(
        """
        UPDATE videos
        SET transcript_path = ?,
            tags_json = ?
        WHERE id = ?
    """,
        (
            str(transcript_path),
            json.dumps(tags, ensure_ascii=False),
            video_id,
        ),
    )

    conn.commit()

    print(f"COMPLETE: {filename}")


# =========================================================
# MAIN
# =========================================================


def scan_videos():
    video_files = []

    for root, dirs, files in os.walk(VIDEOS_DIR):
        for file in files:
            ext = Path(file).suffix.lower()

            if ext in SUPPORTED_VIDEO_EXTENSIONS:
                video_files.append(Path(root) / file)

    return video_files


if __name__ == "__main__":
    start_time = time.time()

    videos = scan_videos()

    print(f"Found {len(videos)} videos")

    for video in videos:
        try:
            process_video(video)

        except Exception as e:
            import traceback
            print(f"ERROR: {video}")
            traceback.print_exc()
    
    elapsed = time.time() - start_time

    print("=" * 60)
    print("ALL COMPLETE")
    print(f"Elapsed: {elapsed / 60:.2f} minutes")
    print("=" * 60)

    conn.close()
