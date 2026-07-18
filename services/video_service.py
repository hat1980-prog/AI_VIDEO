from hashlib import sha1
from pathlib import Path

from config import VIDEOS_DIR


VIDEO_PATTERNS = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.mpg", "*.mpeg")


def get_video_list(scan_directory=None):
    root = Path(scan_directory or VIDEOS_DIR).expanduser()

    if not root.is_dir():
        return []

    root = root.resolve()
    videos = []

    for pattern in VIDEO_PATTERNS:
        videos.extend(root.rglob(pattern))

    videos.sort()

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

    digest = sha1(str(video_path).encode("utf-8")).hexdigest()[:10]
    return f"{video_path.stem}_{digest}"
