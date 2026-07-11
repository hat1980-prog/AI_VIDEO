import os
import glob
import sqlite3
import subprocess

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
import whisper

# =========================
# PATH
# =========================

VIDEOS_DIR = "videos"
FRAMES_DIR = "frames"
FACES_DIR = "faces"
TRANSCRIPTS_DIR = "transcripts"
DB_PATH = "database/metadata.db"

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# =========================
# AI MODELS
# =========================

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

whisper_model = whisper.load_model("base")

# =========================
# DATABASE
# =========================

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

# =========================
# VIDEO LIST
# =========================

def list_videos():
    videos = glob.glob(os.path.join(VIDEOS_DIR, "*.mp4"))
    return [os.path.basename(v) for v in videos]

# =========================
# FRAME EXTRACTION
# =========================

def extract_frames(video_name):

    input_path = os.path.join(VIDEOS_DIR, video_name)

    output_dir = os.path.join(FRAMES_DIR, video_name)
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "%06d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        "fps=1/60",
        output_pattern
    ]

    subprocess.run(cmd)

    count = len(glob.glob(os.path.join(output_dir, "*.jpg")))

    return f"Extracted {count} frames"

# =========================
# FACE EXTRACTION
# =========================

def extract_faces(video_name):

    frame_dir = os.path.join(FRAMES_DIR, video_name)

    face_dir = os.path.join(FACES_DIR, video_name)
    os.makedirs(face_dir, exist_ok=True)

    embeddings = []
    image_paths = []

    frame_files = glob.glob(os.path.join(frame_dir, "*.jpg"))

    total_faces = 0

    for frame_path in frame_files:

        img = cv2.imread(frame_path)

        if img is None:
            continue

        faces = face_app.get(img)

        for idx, face in enumerate(faces):

            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = bbox

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            filename = f"{os.path.basename(frame_path)}_{idx}.jpg"
            save_path = os.path.join(face_dir, filename)

            cv2.imwrite(save_path, crop)

            embeddings.append(face.embedding)
            image_paths.append(save_path)

            total_faces += 1

    if len(embeddings) == 0:
        return "No faces detected"

    embeddings_np = np.array(embeddings)

    clustering = DBSCAN(
        eps=0.6,
        min_samples=3,
        metric='cosine'
    ).fit(embeddings_np)

    labels = clustering.labels_

    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)

    for path, label in zip(image_paths, labels):

        cur.execute(
    """
    INSERT INTO faces
    (video_id, frame_path, face_path, cluster_id, embedding_path)
    VALUES (?, ?, ?, ?, ?)
    """,
        (
            video_id,
            frame_path,
            save_path,
            int(label),
            embedding_file
        )
    )
    conn.commit()

    return f"Faces: {total_faces} / Clusters: {cluster_count}"

# =========================
# TRANSCRIPTION
# =========================

def transcribe_video(video_name):

    video_path = os.path.join(VIDEOS_DIR, video_name)

    result = whisper_model.transcribe(
        video_path,
        language="ja"
    )

    text = result["text"]

    output_path = os.path.join(
        TRANSCRIPTS_DIR,
        video_name + ".txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    cur.execute(
        "INSERT INTO videos(filename, transcript, tags) VALUES (?, ?, ?)",
        (video_name, text, "")
    )

    conn.commit()

    return text

# =========================
# CLUSTER VIEW
# =========================

def show_clusters():

    cur.execute(
        "SELECT cluster_id, image_path FROM faces WHERE cluster_id >= 0"
    )

    rows = cur.fetchall()

    data = []

    for cluster_id, image_path in rows:
        data.append({
            "cluster": cluster_id,
            "image": image_path
        })

    df = pd.DataFrame(data)

    return df

# =========================
# UI
# =========================

with gr.Blocks() as demo:

    gr.Markdown("# AI Video Analyzer")

    with gr.Row():

        video_dropdown = gr.Dropdown(
            choices=list_videos(),
            label="Video"
        )

    with gr.Row():

        extract_btn = gr.Button("1. Extract Frames")
        face_btn = gr.Button("2. Extract Faces")
        transcribe_btn = gr.Button("3. Transcribe")

    output_text = gr.Textbox(
        label="Output",
        lines=10
    )

    extract_btn.click(
        extract_frames,
        inputs=video_dropdown,
        outputs=output_text
    )

    face_btn.click(
        extract_faces,
        inputs=video_dropdown,
        outputs=output_text
    )

    transcribe_btn.click(
        transcribe_video,
        inputs=video_dropdown,
        outputs=output_text
    )

    cluster_btn = gr.Button("Show Clusters")

    cluster_table = gr.Dataframe()

    cluster_btn.click(
        show_clusters,
        outputs=cluster_table
    )

# =========================
# START
# =========================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)