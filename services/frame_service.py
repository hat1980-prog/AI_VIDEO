from pathlib import Path
import json
import subprocess
import shutil
import time

from config import FRAMES_DIR
from config import FRAME_INTERVAL
from config import FRAME_KEYFRAMES_ONLY
from config import FFMPEG_STALL_TIMEOUT_SECONDS
from services.video_service import get_video_id, get_video_path
from services.runtime_log_service import append_runtime_log


def _ffmpeg_error_preview(stderr):
    lines = [line for line in stderr.splitlines() if line.strip()]
    return "\n".join(lines[:10])


def _summarize_ffmpeg_error(stderr):
    preview = _ffmpeg_error_preview(stderr)
    line_count = len([line for line in stderr.splitlines() if line.strip()])

    if "Invalid NAL unit size" in stderr or "Error splitting the input into NAL units" in stderr:
        return (
            "[ERROR] H.264映像ストリームの破損または不正なパケットを検出しました。"
            "この動画は正常にデコードできないため、抽出を停止しました。\n"
            f"FFmpegエラー（先頭{min(line_count, 10)}行）:\n{preview}"
        )

    return (
        f"[ERROR] FFmpegエラー（先頭{min(line_count, 10)}行）:\n{preview}"
        if preview else "[ERROR] FFmpegエラーが発生しました"
    )


def _stop_ffmpeg(process):
    """Terminate FFmpeg without leaving a stuck child process behind."""
    process.terminate()
    try:
        return process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        append_runtime_log("[FFmpeg] 終了待機がタイムアウトしたため強制終了します")
        process.kill()
        return process.communicate()


def _run_ffmpeg(command, should_cancel, output_dir=None, stall_timeout_seconds=None):
    append_runtime_log("[FFmpeg] 実行コマンド: " + " ".join(map(str, command)))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started_at = time.perf_counter()
    next_status_at = started_at + 5.0
    last_frame_count = 0
    last_output_at = started_at

    while process.poll() is None:
        if should_cancel and should_cancel():
            _, stderr = _stop_ffmpeg(process)
            append_runtime_log("[FFmpeg] [CANCELLED] 終了コード: " + str(process.returncode))
            return "cancelled", process.returncode, stderr
        now = time.perf_counter()
        if output_dir is not None and now >= next_status_at:
            frame_count = sum(1 for _ in output_dir.glob("*.jpg"))
            if frame_count > last_frame_count:
                last_frame_count = frame_count
                last_output_at = now
            silent_seconds = now - last_output_at
            append_runtime_log(
                f"[FFmpeg] 実行中: {now - started_at:.0f}秒経過 / "
                f"抽出済みフレーム: {frame_count}枚 / 最終出力から: {silent_seconds:.0f}秒"
            )
            if stall_timeout_seconds is not None and silent_seconds >= stall_timeout_seconds:
                append_runtime_log(
                    "[FFmpeg] [TIMEOUT] フレーム出力が "
                    f"{silent_seconds:.0f}秒間増加しないため、この動画の抽出を打ち切ります"
                )
                _, stderr = _stop_ffmpeg(process)
                return "stalled", process.returncode, stderr
            next_status_at = now + 5.0
        time.sleep(0.1)

    _, stderr = process.communicate()
    result = f"[FFmpeg] 終了コード: {process.returncode}"
    if stderr.strip():
        result += " / メッセージ: " + _ffmpeg_error_preview(stderr).replace("\n", " | ")
    append_runtime_log(result)
    return None, process.returncode, stderr


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
    # A long sampling interval can legitimately produce no new image for a
    # while.  Leave room for two intervals, while still providing a hard
    # fail-safe for a decoder/process that is stuck indefinitely.
    stall_timeout_seconds = max(FFMPEG_STALL_TIMEOUT_SECONDS, interval * 2)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        # Continue decoding after corrupt H.264 packets when possible.  Using
        # -xerror here made one invalid NAL unit abort the entire video.
        "-fflags",
        "+discardcorrupt",
        "-err_detect",
        "ignore_err",
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
    stop_reason, returncode, stderr = _run_ffmpeg(
        command, should_cancel, output_dir, stall_timeout_seconds
    )
    if stop_reason == "cancelled":
        elapsed = time.perf_counter() - started_at
        return False, (
            f"[CANCELLED] フレーム抽出を中断しました（{elapsed:.1f}秒）\n"
            f"{_summarize_ffmpeg_error(stderr)}"
        )
    if stop_reason == "stalled":
        elapsed = time.perf_counter() - started_at
        return False, (
            "[TIMEOUT] FFmpegのフレーム出力が "
            f"{stall_timeout_seconds}秒間増加しなかったため、この動画の抽出を停止しました"
            f"（経過時間: {elapsed:.1f}秒）。次の動画の解析へ進みます。"
        )

    fallback_used = False
    if returncode != 0 and FRAME_KEYFRAMES_ONLY:
        # Some playable files fail when FFmpeg decodes keyframes only. Retry
        # once with ordinary decoding before accepting a partial extraction.
        fallback_used = True
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        fallback_command = [item for item in command if item not in {"-skip_frame", "nokey"}]
        flags_index = fallback_command.index("-fflags") + 1
        fallback_command[flags_index] = "+genpts+discardcorrupt"
        stop_reason, returncode, fallback_stderr = _run_ffmpeg(
            fallback_command, should_cancel, output_dir, stall_timeout_seconds
        )
        if stop_reason == "cancelled":
            elapsed = time.perf_counter() - started_at
            return False, (
                f"[CANCELLED] フレーム抽出を中断しました（{elapsed:.1f}秒）\n"
                f"{_summarize_ffmpeg_error(fallback_stderr)}"
            )
        if stop_reason == "stalled":
            elapsed = time.perf_counter() - started_at
            return False, (
                "[TIMEOUT] 通常デコードへの再試行中もFFmpegのフレーム出力が "
                f"{stall_timeout_seconds}秒間増加しなかったため、この動画の抽出を停止しました"
                f"（経過時間: {elapsed:.1f}秒）。次の動画の解析へ進みます。"
            )
        stderr = f"{stderr}\n{fallback_stderr}".strip()

    count = len(list(output_dir.glob("*.jpg")))
    elapsed = time.perf_counter() - started_at

    if returncode != 0 and count == 0:
        return False, f"[ERROR] フレーム抽出に失敗しました（{elapsed:.1f}秒）\n{_summarize_ffmpeg_error(stderr)}"

    (output_dir / "frames_metadata.json").write_text(
        json.dumps(
            {
                "frame_interval_seconds": interval,
                "timestamp_basis": "sample_index",
                "keyframes_only": FRAME_KEYFRAMES_ONLY and not fallback_used,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    mode = "キーフレーム優先" if FRAME_KEYFRAMES_ONLY else "全フレーム"
    if fallback_used:
        mode += "（通常デコードへ自動再試行）"
    skipped_error_message = ""
    if stderr.strip():
        skipped_error_message = (
            "\n[WARNING] FFmpegメッセージを検出しました。"
            "破損・不正な映像パケットはスキップして、抽出可能な範囲を処理しました。"
            f"\nFFmpegメッセージ（先頭10行）:\n{_ffmpeg_error_preview(stderr)}"
        )
    recovery_message = "\n[RECOVERY] キーフレーム優先抽出に失敗したため、通常デコードで再試行しました。" if fallback_used else ""
    return True, (
        f"[OK] {count} 枚抽出しました"
        f"（{interval}秒ごと、{mode}、所要時間: {elapsed:.1f}秒）"
        f"{recovery_message}"
        f"{skipped_error_message}"
    )
