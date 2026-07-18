from pathlib import Path
import subprocess
import shutil

from config import FRAMES_DIR
from config import FRAME_FPS
from services.video_service import get_video_id, get_video_path


def extract_frames(video_name):

    video_path = get_video_path(video_name)

    if video_path is None:
        return False, "動画ファイルが見つかりません"

    output_dir = FRAMES_DIR / get_video_id(video_name)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)

    output_pattern = output_dir / "%06d.jpg"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={FRAME_FPS}",
        str(output_pattern)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return False, result.stderr

    count = len(list(output_dir.glob("*.jpg")))

    return True, f"{count} 枚抽出しました"
