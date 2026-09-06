import gradio as gr
from time import perf_counter

from services.frame_service import extract_frames
from services.face_service import extract_faces
from services.cluster_service import cluster_faces
from services.gallery_service import load_gallery
from services.cancellation_service import (
    begin_analysis,
    consume_current_video_skip,
    is_analysis_cancelled,
    is_current_video_skip_requested,
    should_stop_current_video,
)
from services.video_data_service import clear_video_data_for_reanalysis
from services.runtime_log_service import append_runtime_log, clear_runtime_log


def analyze_video(video_name, frame_interval=None, progress=gr.Progress(), reset_cancel=True):

    if not video_name:
        return "動画を選択してください", []

    if reset_cancel:
        begin_analysis()
        clear_runtime_log()

    started_at = perf_counter()
    logs = ["=== Analyze開始 ==="]
    append_runtime_log(f"[Analyze] 開始: {video_name}")

    progress(0.02, desc="前回の解析データを整理しています")
    append_runtime_log("[0/3] 再解析前データを整理中")
    cleanup_message = clear_video_data_for_reanalysis(video_name)
    logs.append(f"[0/3 再解析データの整理] [OK]\n{cleanup_message}")

    progress(0.05, desc="フレームを抽出しています")
    append_runtime_log("[1/3] フレーム抽出を開始")

    phase_started_at = perf_counter()
    ok, msg = extract_frames(video_name, frame_interval, should_stop_current_video)
    frame_status = (
        "[ERROR]" if not ok
        else "[WARNING]" if "[WARNING]" in msg
        else "[RECOVERY]" if "[RECOVERY]" in msg
        else "[OK]"
    )
    logs.append(f"[1/3 フレーム抽出] {frame_status} {perf_counter() - phase_started_at:.1f}秒\n{msg}")

    if not ok:
        append_runtime_log("[1/3] [ERROR] フレーム抽出を終了")
        skipped = consume_current_video_skip()
        progress(1.0, desc="この動画をスキップしました" if skipped else ("解析を中断しました" if is_analysis_cancelled() else "フレーム抽出に失敗しました"))
        if skipped:
            logs.append("スキップ要求を受けたため、この動画の解析を終了して次の動画へ進みます")
        logs.append(f"Analyze合計: {perf_counter() - started_at:.1f}秒")
        return "\n\n".join(logs), []

    progress(0.25, desc="顔を検出しています（0%）")
    append_runtime_log("[2/3] 顔抽出・照合を開始")

    def update_face_progress(value):
        percentage = int(value * 100)
        progress(
            0.25 + value * 0.60,
            desc=f"顔を検出しています（{percentage}%）"
        )

    phase_started_at = perf_counter()
    face_message = extract_faces(
        video_name,
        progress_callback=update_face_progress,
        should_cancel=should_stop_current_video,
        frame_interval=frame_interval,
    )
    logs.append(f"[2/3 顔抽出・照合] [OK] {perf_counter() - phase_started_at:.1f}秒\n{face_message}")

    if should_stop_current_video():
        append_runtime_log("[2/3] [CANCELLED] 顔抽出・照合を終了")
        skipped = consume_current_video_skip()
        progress(1.0, desc="この動画をスキップしました" if skipped else "解析を中断しました")
        if skipped:
            logs.append("スキップ要求を受けたため、この動画の解析を終了して次の動画へ進みます")
        logs.append(f"Analyze合計: {perf_counter() - started_at:.1f}秒")
        return "\n\n".join(logs), []

    progress(0.90, desc="人物をクラスタリングしています")
    append_runtime_log("[3/3] 人物クラスタリングを開始")
    phase_started_at = perf_counter()
    cluster_message = cluster_faces(video_name, should_stop_current_video)
    logs.append(f"[3/3 人物クラスタリング] [OK] {perf_counter() - phase_started_at:.1f}秒\n{cluster_message}")

    if should_stop_current_video():
        append_runtime_log("[3/3] [CANCELLED] 人物クラスタリングを終了")
        skipped = consume_current_video_skip()
        progress(1.0, desc="この動画をスキップしました" if skipped else "解析を中断しました")
        if skipped:
            logs.append("スキップ要求を受けたため、この動画の解析を終了して次の動画へ進みます")
        logs.append(f"Analyze合計: {perf_counter() - started_at:.1f}秒")
        return "\n\n".join(logs), []

    progress(0.98, desc="結果を表示しています")
    gallery = load_gallery(video_name)

    progress(1.0, desc="解析が完了しました")
    logs.append(f"Analyze合計: {perf_counter() - started_at:.1f}秒")
    append_runtime_log(f"[Analyze] [OK] 完了: {video_name} / {perf_counter() - started_at:.1f}秒")

    return "\n\n".join(logs), gallery
