import shutil
import sqlite3
from pathlib import Path

BASE_DIR = Path(r"B:\AI_VIDEO")

DB_PATH = BASE_DIR / "database" / "metadata.db"

ACTORS_DIR = BASE_DIR / "actors"

ACTORS_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

query = """
SELECT
    cluster_id,
    face_path
FROM faces
WHERE cluster_id != -1
"""

rows = cursor.execute(query).fetchall()

clusters = {}

for cluster_id, face_path in rows:

    clusters.setdefault(cluster_id, [])

    clusters[cluster_id].append(face_path)

print(f"Found {len(clusters)} actors")

for cluster_id, face_paths in clusters.items():

    actor_dir = ACTORS_DIR / f"actor_{cluster_id:04d}"

    faces_dir = actor_dir / "faces"

    faces_dir.mkdir(parents=True, exist_ok=True)

    for idx, face_path in enumerate(face_paths):

        src = Path(face_path)

        if not src.exists():
            continue

        dst = faces_dir / f"{idx:05d}.jpg"

        shutil.copy(src, dst)


import cv2

for cluster_id in clusters.keys():

    actor_dir = ACTORS_DIR / f"actor_{cluster_id:04d}"

    faces_dir = actor_dir / "faces"

    jpgs = list(faces_dir.glob("*.jpg"))

    best_img = None
    best_area = 0

    for jpg in jpgs:

        img = cv2.imread(str(jpg))

        if img is None:
            continue

        h, w = img.shape[:2]

        area = h * w

        if area > best_area:
            best_area = area
            best_img = img

    if best_img is not None:

        cv2.imwrite(
            str(actor_dir / "thumbnail.jpg"),
            best_img
        )


print("COMPLETE")