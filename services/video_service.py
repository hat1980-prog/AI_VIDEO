from pathlib import Path

from config import VIDEOS_DIR


def get_video_list():

    videos = []

    for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov"):

        videos.extend(VIDEOS_DIR.glob(ext))

    videos.sort()

    return [v.name for v in videos]