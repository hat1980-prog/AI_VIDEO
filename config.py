from pathlib import Path

BASE_DIR = Path(__file__).parent

VIDEOS_DIR = BASE_DIR / "videos"
FRAMES_DIR = BASE_DIR / "frames"
FACES_DIR = BASE_DIR / "faces"
DATABASE_DIR = BASE_DIR / "database"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"

VIDEOS_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# 何秒ごとにフレームを抽出するか
FRAME_INTERVAL = 2

FRAME_FPS = f"1/{FRAME_INTERVAL}"

FACE_MODEL = "buffalo_l"

# GPU:0 / CPU:-1
FACE_CTX = 0

EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# -----------------------------
# 人物ライブラリ
# -----------------------------
PERSON_LIBRARY_DIR = BASE_DIR / "person_library"
PERSON_LIBRARY_DIR.mkdir(exist_ok=True)

ACTORS_DIR = BASE_DIR / "actors"
ACTORS_DIR.mkdir(exist_ok=True)