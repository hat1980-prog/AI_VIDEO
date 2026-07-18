import gradio as gr

from services.frame_service import extract_frames
from services.face_service import extract_faces
from services.cluster_service import cluster_faces
from services.gallery_service import load_gallery


def analyze_video(video_name, progress=gr.Progress()):

    if not video_name:
        return "動画を選択してください", []

    logs = []

    progress(0.05, desc="フレームを抽出しています")

    ok, msg = extract_frames(video_name)
    logs.append(msg)

    if not ok:
        progress(1.0, desc="フレーム抽出に失敗しました")
        return "\n".join(logs), []

    progress(0.25, desc="顔を検出しています（0%）")

    def update_face_progress(value):
        percentage = int(value * 100)
        progress(
            0.25 + value * 0.60,
            desc=f"顔を検出しています（{percentage}%）"
        )

    logs.append(extract_faces(video_name, progress_callback=update_face_progress))

    progress(0.90, desc="人物をクラスタリングしています")
    logs.append(cluster_faces(video_name))

    progress(0.98, desc="結果を表示しています")
    gallery = load_gallery(video_name)

    progress(1.0, desc="解析が完了しました")

    return "\n".join(logs), gallery
