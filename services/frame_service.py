from pathlib import Path
import json
import subprocess
import shutil
import time

from config import FRAMES_DIR
from config import FRAME_INTERVAL
from config import FRAME_KEYFRAMES_ONLY
from services.video_service import get_video_id, get_video_path


def _summarize_ffmpeg_error(stderr):
    lines = [line for line in stderr.splitlines() if line.strip()]
    preview = "\n".join(lines[:10])

    if "Invalid NAL unit size" in stderr or "Error splitting the input into NAL units" in stderr:
        return (
            "H.264映像ストリームの破損または不正なパケットを検出しました。"
            "この動画は正常にデコードできないため、抽出を停止しました。\n"
            f"FFmpegエラー（先頭{min(len(lines), 10)}行）:\n{preview}"
        )

    return f"FFmpegエラー（先頭{min(len(lines), 10)}行）:\n{preview}" if preview else "FFmpegエラーが発生しました"


def extract_frames(video_name, frame_interval=None, should_cancel=None):

    video_path = get_video_path(video_name)

    if video_path is None:
        return False, "動画ファイルが見つかりません"

    try:
        interval = int(frame_interval if frame_interval is not None else FRAME_INTERVAL)
    except (TypeError, ValueError):
        return False, "フレーム抽出間隔は1秒以上の整数で指定してください"

    if interval < 1:
        return False, "フレーム抽出間隔は1秒以上で指定してください"

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
        "-xerror",
        "-y",
    ]

    if FRAME_KEYFRAMES_ONLY:
        command.extend(["-skip_frame", "nokey"])

    command.extend([
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval}:round=down",
        str(output_pattern)
    ])

    started_at = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while process.poll() is None:
        if should_cancel and should_cancel():
            process.terminate()
            _, stderr = process.communicate()
            elapsed = time.perf_counter() - started_at
            return False, (
                f"フレーム抽出を中断しました（{elapsed:.1f}秒）\n"
                f"{_summarize_ffmpeg_error(stderr)}"
            )

        time.sleep(0.1)

    _, stderr = process.communicate()

    if process.returncode != 0:
        elapsed = time.perf_counter() - started_at
        return False, f"フレーム抽出に失敗しました（{elapsed:.1f}秒）\n{_summarize_ffmpeg_error(stderr)}"

    count = len(list(output_dir.glob("*.jpg")))
    elapsed = time.perf_counter() - started_at

    (output_dir / "frames_metadata.json").write_text(
        json.dumps(
            {
                "frame_interval_seconds": interval,
                "timestamp_basis": "sample_index",
                "keyframes_only": FRAME_KEYFRAMES_ONLY,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    mode = "キーフレーム優先" if FRAME_KEYFRAMES_ONLY else "全フレーム"
    return True, (
        f"{count} 枚抽出しました"
        f"（{interval}秒ごと、{mode}、所要時間: {elapsed:.1f}秒）"
    )
