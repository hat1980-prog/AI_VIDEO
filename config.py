from pathlib import Path

BASE_DIR = Path(__file__).parent

VIDEOS_DIR = BASE_DIR / "videos"
FRAMES_DIR = BASE_DIR / "frames"
FACES_DIR = BASE_DIR / "faces"
DATABASE_DIR = BASE_DIR / "database"
VIDEO_IDENTITY_INDEX_PATH = DATABASE_DIR / "video_identity_index.json"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"

VIDEOS_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# 何秒ごとにフレームを抽出するか
FRAME_INTERVAL = 180

FRAME_FPS = f"1/{FRAME_INTERVAL}"

# 長尺動画の顔サンプリングではキーフレームだけをデコードして高速化する
FRAME_KEYFRAMES_ONLY = True

# FFmpeg が終了せず、フレーム出力も止まった場合に動画単位で打ち切るまでの秒数。
# 実際の監視時間は、長いサンプリング間隔を考慮して
# max(この値, FRAME_INTERVAL の 2 倍) となる。
FFMPEG_STALL_TIMEOUT_SECONDS = 300

FACE_MODEL = "buffalo_l"

# GPU:0 / CPU:-1
FACE_CTX = 0

# 保存する顔画像の最小サイズ（短辺・px）
MIN_FACE_SIZE = 128

# 人物クラスタリングの許容コサイン距離（大きいほど同一人物にまとめやすい）
CLUSTER_EPS = 0.65

# 出演者ライブラリへの自動紐付けに使う類似度しきい値
PERSON_SIMILARITY_THRESHOLD = 0.65
PERSON_SIMILARITY_THRESHOLD_MIN = 0.45
PERSON_SIMILARITY_THRESHOLD_MAX = 0.95
PERSON_SIMILARITY_MARGIN = 0.03
# 一括再分類で出演者ごとに参照する個別学習顔Embeddingの最大数
RECLASSIFICATION_FACE_EXEMPLARS = 8
PERSON_LIBRARY_SETTINGS_PATH = DATABASE_DIR / "person_library_settings.json"

# 除外済み人物との照合設定
EXCLUDED_FACES_DIR = BASE_DIR / "excluded_faces"
EXCLUDED_FACES_DIR.mkdir(exist_ok=True)
EXCLUDED_FACE_SIMILARITY_THRESHOLD = 0.65

EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# -----------------------------
# 人物ライブラリ
# -----------------------------
PERSON_LIBRARY_DIR = BASE_DIR / "person_library"
PERSON_LIBRARY_DIR.mkdir(exist_ok=True)

ACTORS_DIR = BASE_DIR / "actors"
ACTORS_DIR.mkdir(exist_ok=True)
