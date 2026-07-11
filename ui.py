import gradio as gr

from services.video_service import get_video_list
from services.frame_service import extract_frames
from services.face_service import extract_faces
from services.cluster_service import cluster_faces
from services.gallery_service import load_gallery
from services.analyze_service import analyze_video
from services.actor_service import (
    load_actor_gallery,
    load_actor_list
)

def refresh():

    return gr.update(
        choices=get_video_list()
    )


def selected(video):

    if not video:
        return "動画を選択してください"

    return f"選択中：{video}"


def cluster_and_gallery(video):

    message = cluster_faces(video)

    gallery = load_gallery(video)

    return message, gallery


def create_ui():

    with gr.Blocks(title="AI Video Analyzer") as demo:

        gr.Markdown("# AI Video Analyzer")

        video = gr.Dropdown(
            choices=get_video_list(),
            label="動画"
        )

        with gr.Row():

            refresh_btn = gr.Button("更新")

            frame_btn = gr.Button("Extract Frames")

            face_btn = gr.Button("Extract Faces")

            cluster_btn = gr.Button("Cluster Faces")

            analyze_btn = gr.Button("Analyze", variant="primary")

        result = gr.Textbox(
            label="ログ",
            lines=15
        )

        gallery = gr.Gallery(
            label="人物一覧",
            columns=5,
            rows=2,
            height=300,
            object_fit="contain",
            preview=True
        )

        gr.Markdown("## 出演者ライブラリ")

        refresh_actor_btn = gr.Button("出演者更新")

        actor_gallery = gr.Gallery(
            label="出演者",
            columns=5,
            rows=2,
            height=300,
            object_fit="contain",
            preview=True
        )

        refresh_btn.click(
            refresh,
            outputs=video
        )

        video.change(
            selected,
            video,
            result
        )

        frame_btn.click(
            extract_frames,
            video,
            result
        )

        face_btn.click(
            extract_faces,
            video,
            result
        )

        cluster_btn.click(
            fn=cluster_and_gallery,
            inputs=video,
            outputs=[
                result,
                gallery
            ]
        )

        analyze_btn.click(
            fn=analyze_video,
            inputs=video,
            outputs=[
                result,
                gallery
            ]
        )

        refresh_actor_btn.click(
            fn=load_actor_gallery,
            outputs=actor_gallery
        )

    return demo