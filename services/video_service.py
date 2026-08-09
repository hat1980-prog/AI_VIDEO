from hashlib import sha1
from pathlib import Path

from config import EMBEDDINGS_DIR, FACES_DIR, VIDEOS_DIR


VIDEO_PATTERNS = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.mpg", "*.mpeg", "*.wmv", "*.ts")


def _has_analysis_data(video_path):
    video_id = _get_video_id_from_path(video_path)
    return (FACES_DIR / video_id).exists() or (EMBEDDINGS_DIR / video_id).exists()


def _get_video_id_from_path(video_path):
    digest = sha1(str(video_path).encode("utf-8")).hexdigest()[:10]
    return f"{video_path.stem}_{digest}"


def get_video_list(scan_directory=None, exclude_analyzed=False):
    root = Path(scan_directory or VIDEOS_DIR).expanduser()

    if not root.is_dir():
        return []

    root = root.resolve()
    videos = []

    for pattern in VIDEO_PATTERNS:
        videos.extend(root.rglob(pattern))

    videos.sort()

    if exclude_analyzed:
        videos = [video for video in videos if not _has_analysis_data(video)]

    return [
        (str(video.relative_to(root)), str(video.resolve()))
        for video in videos
    ]


def get_video_path(video):
    if not video:
        return None

    video_path = Path(video).expanduser()

    if video_path.is_file():
        return video_path.resolve()

    default_path = VIDEOS_DIR / video_path

    if default_path.is_file():
        return default_path.resolve()

    return None


def get_video_id(video):
    video_path = get_video_path(video)

    if video_path is None:
        return None

    return _get_video_id_from_path(video_path)
