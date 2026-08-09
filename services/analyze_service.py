import gradio as gr
from time import perf_counter

from services.frame_service import extract_frames
from services.face_service import extract_faces
from services.cluster_service import cluster_faces
from services.gallery_service import load_gallery
from services.cancellation_service import begin_analysis, is_analysis_cancelled


def analyze_video(video_name, frame_interval=None, progress=gr.Progress(), reset_cancel=True):

    if not video_name:
        return "動画を選択してください", []

    if reset_cancel:
        begin_analysis()

    started_at = perf_counter()
    logs = ["=== Analyze開始 ==="]

    progress(0.05, desc="フレームを抽出しています")

    phase_started_at = perf_counter()
    ok, msg = extract_frames(video_name, frame_interval, is_analysis_cancelled)
    logs.append(f"[1/3 フレーム抽出] {perf_counter() - phase_started_at:.1f}秒\n{msg}")

    if not ok:
        progress(1.0, desc="解析を中断しました" if is_analysis_cancelled() else "フレーム抽出に失敗しました")
        logs.append(f"Analyze合計: {perf_counter() - started_at:.1f}秒")
        return "\n\n".join(logs), []

    progress(0.25, desc="顔を検出しています（0%）")

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
        should_cancel=is_analysis_cancelled,
        frame_interval=frame_interval,
    )
    logs.append(f"[2/3 顔抽出・照合] {perf_counter() - phase_started_at:.1f}秒\n{face_message}")

    if is_analysis_cancelled():
        progress(1.0, desc="解析を中断しました")
        logs.append(f"Analyze合計: {perf_counter() - started_at:.1f}秒")
        return "\n\n".join(logs), load_gallery(video_name)

    progress(0.90, desc="人物をクラスタリングしています")
    phase_started_at = perf_counter()
    cluster_message = cluster_faces(video_name, is_analysis_cancelled)
    logs.append(f"[3/3 人物クラスタリング] {perf_counter() - phase_started_at:.1f}秒\n{cluster_message}")

    if is_analysis_cancelled():
        progress(1.0, desc="解析を中断しました")
        logs.append(f"Analyze合計: {perf_counter() - started_at:.1f}秒")
        return "\n\n".join(logs), load_gallery(video_name)

    progress(0.98, desc="結果を表示しています")
    gallery = load_gallery(video_name)

    progress(1.0, desc="解析が完了しました")
    logs.append(f"Analyze合計: {perf_counter() - started_at:.1f}秒")

    return "\n\n".join(logs), gallery
