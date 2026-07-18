from pathlib import Path

import gradio as gr

from config import VIDEOS_DIR
from services.video_service import get_video_list
from services.database_viewer_service import get_database_tables, get_table_data
from services.exclusion_service import exclude_face
from services.person_registration_service import register_cluster_as_actor
from services.frame_service import extract_frames
from services.face_service import extract_faces
from services.cluster_service import cluster_faces
from services.gallery_service import load_gallery
from services.analyze_service import analyze_video
from services.actor_service import (
    delete_actor,
    get_actor_details,
    get_similar_actor_choices,
    load_actor_library,
    load_actor_list,
    merge_actors,
    rename_actor,
)
from services.reset_service import reset_actor_library, reset_database


def refresh(scan_directory, progress=gr.Progress()):
    progress(0.1, desc="動画一覧を取得しています")
    choices = get_video_list(scan_directory)
    progress(1.0, desc="動画一覧を更新しました")
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=[]),
        f"{len(choices)}件の動画を検出しました",
    )


def selected(video):
    if not video:
        return "動画を選択してください"

    return f"選択中：{Path(video).name}"


def refresh_database_viewer(progress=gr.Progress()):
    progress(0.1, desc="データベースを読み込んでいます")
    tables = get_database_tables()

    if not tables:
        progress(1.0, desc="データベースにテーブルがありません")
        return (
            gr.update(choices=[], value=None),
            gr.update(value=[[""]], headers=["データなし"], column_count=1),
            "表示できるテーブルがありません",
        )

    headers, rows, message = get_table_data(tables[0])
    progress(1.0, desc="データベースを読み込みました")
    return (
        gr.update(choices=tables, value=tables[0]),
        gr.update(value=rows, headers=headers, column_count=len(headers)),
        message,
    )


def select_database_table(table_name, progress=gr.Progress()):
    progress(0.1, desc="テーブルを読み込んでいます")
    headers, rows, message = get_table_data(table_name)
    progress(1.0, desc="テーブルを読み込みました")
    return gr.update(value=rows, headers=headers, column_count=len(headers)), message


def analyze_multiple_videos(videos, progress=gr.Progress()):
    if not videos:
        return "連続Analyzeの対象動画を選択してください", []

    logs = []
    gallery = []
    total = len(videos)

    for index, video in enumerate(videos):
        video_name = Path(video).name

        def update_progress(value, desc):
            percentage = (index + value) / total
            progress(percentage, desc=f"{index + 1}/{total}: {video_name} - {desc}")

        update_progress(0.0, "解析を開始しています")
        result, video_gallery = analyze_video(video, progress=update_progress)
        logs.append(f"=== {video_name} ===\n{result}")
        gallery.extend(video_gallery)

    progress(1.0, desc="選択したすべての動画の解析が完了しました")
    return "\n\n".join(logs), gallery


def extract_frames_with_progress(video, progress=gr.Progress()):
    progress(0.05, desc="フレーム抽出を開始しています")
    ok, message = extract_frames(video)
    progress(1.0, desc="フレーム抽出が完了しました" if ok else "フレーム抽出に失敗しました")
    return message


def extract_faces_with_progress(video, progress=gr.Progress()):
    progress(0.05, desc="顔検出を開始しています")

    def update_progress(value):
        progress(value, desc=f"顔を検出しています（{int(value * 100)}%）")

    message = extract_faces(video, progress_callback=update_progress)
    progress(1.0, desc="顔検出が完了しました")
    return message


def cluster_and_gallery(video, progress=gr.Progress()):
    progress(0.05, desc="人物をクラスタリングしています")
    message = cluster_faces(video)
    progress(0.9, desc="人物一覧を作成しています")
    gallery = load_gallery(video)
    progress(1.0, desc="クラスタリングが完了しました")
    return message, gallery


def select_detected_face(video, evt: gr.SelectData):
    faces = load_gallery(video)

    if evt.index is None or evt.index >= len(faces):
        return (
            None,
            None,
            "",
            gr.update(choices=[], value=None),
            "顔画像を選択してください",
        )

    face_path, label = faces[evt.index]
    actor_name = "" if label == "Unknown" else label
    selected_cluster = label if label.startswith("Person_") else None

    return (
        face_path,
        selected_cluster,
        actor_name,
        gr.update(choices=load_actor_list(), value=None),
        f"選択中：{Path(face_path).name}（{label}）",
    )


def exclude_selected_face(video, face_path, progress=gr.Progress()):
    progress(0.1, desc="顔画像を除外しています")
    message = exclude_face(video, face_path)
    progress(0.85, desc="顔一覧を更新しています")
    gallery = load_gallery(video)
    progress(1.0, desc="除外処理が完了しました")
    return None, gallery, "顔画像を選択してください", message


def register_selected_cluster(video, cluster_name, actor_name, merge_target, progress=gr.Progress()):
    progress(0.1, desc="出演者ライブラリへ登録しています")
    target_actor_name = merge_target or actor_name
    message = register_cluster_as_actor(video, cluster_name, target_actor_name)
    progress(0.85, desc="出演者ライブラリを更新しています")
    actor_gallery, actor_count = load_actor_library()
    progress(1.0, desc="出演者登録が完了しました")
    return actor_gallery, actor_count, message


def select_actor(sort_by, sort_order, evt: gr.SelectData):
    gallery, _ = load_actor_library(sort_by, sort_order)

    if evt.index is None or evt.index >= len(gallery):
        return (
            None,
            "",
            [],
            "出演者を選択してください",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
        )

    actor_name = gallery[evt.index][1]
    name, faces, detail = get_actor_details(actor_name)
    choices = get_similar_actor_choices(actor_name)

    return (
        actor_name,
        name,
        faces,
        detail,
        gr.update(choices=choices, value=None),
        gr.update(choices=[], value=None),
    )


def refresh_actor_library(sort_by, sort_order, progress=gr.Progress()):
    progress(0.1, desc="出演者ライブラリを更新しています")
    gallery, count = load_actor_library(sort_by, sort_order)
    progress(1.0, desc="出演者ライブラリを更新しました")
    return gallery, count


def rename_selected_actor(actor_name, new_name, sort_by, sort_order, progress=gr.Progress()):
    progress(0.1, desc="出演者名を変更しています")
    selected_name, message = rename_actor(actor_name, new_name)
    gallery, count = load_actor_library(sort_by, sort_order)
    name, faces, detail = get_actor_details(selected_name)
    choices = get_similar_actor_choices(selected_name)
    progress(1.0, desc="出演者名の変更が完了しました")

    return (
        selected_name,
        gallery,
        count,
        name,
        faces,
        detail,
        gr.update(choices=choices, value=None),
        message,
    )


def delete_selected_actor(actor_name, sort_by, sort_order, progress=gr.Progress()):
    progress(0.1, desc="出演者を削除しています")
    message = delete_actor(actor_name)
    gallery, count = load_actor_library(sort_by, sort_order)

    progress(1.0, desc="出演者の削除が完了しました")

    return (
        None,
        gallery,
        count,
        "",
        [],
        "出演者を選択してください",
        gr.update(choices=[], value=None),
        message,
    )


def update_merge_result_name(source_name, target_name):
    choices = [name for name in (source_name, target_name) if name]
    return gr.update(choices=choices, value=target_name or None)


def merge_selected_actors(source_name, target_name, result_name, sort_by, sort_order, progress=gr.Progress()):
    progress(0.05, desc="出演者を統合しています")
    message = merge_actors(source_name, target_name, result_name)
    progress(0.85, desc="出演者ライブラリを更新しています")
    gallery, count = load_actor_library(sort_by, sort_order)
    progress(1.0, desc="出演者の統合と再分類が完了しました")

    return (
        None,
        gallery,
        count,
        "",
        [],
        "出演者を選択してください",
        gr.update(choices=[], value=None),
        message,
    )


def initialize_database(confirmed, progress=gr.Progress()):
    if not confirmed:
        return False, "初期化するには確認チェックを入れてください"

    progress(0.1, desc="データベースを初期化しています")
    message = reset_database()
    progress(1.0, desc="データベースの初期化が完了しました")
    return False, message


def initialize_actor_library(confirmed, progress=gr.Progress()):
    if not confirmed:
        return (
            False, None, *load_actor_library(), "", [],
            "出演者を選択してください", gr.update(choices=[], value=None),
            "初期化するには確認チェックを入れてください"
        )

    progress(0.1, desc="出演者ライブラリを初期化しています")
    message = reset_actor_library()
    progress(1.0, desc="出演者ライブラリの初期化が完了しました")

    return (
        False, None, [], "出演者数：0", "", [],
        "出演者を選択してください", gr.update(choices=[], value=None), message
    )


def create_ui():
    with gr.Blocks(title="AI Video Analyzer") as demo:
        gr.Markdown("# AI Video Analyzer")

        scan_directory = gr.Textbox(label="動画スキャンフォルダ", value=str(VIDEOS_DIR))
        video = gr.Dropdown(choices=get_video_list(VIDEOS_DIR), label="動画")
        batch_videos = gr.CheckboxGroup(
            choices=get_video_list(VIDEOS_DIR),
            label="連続Analyze対象（複数選択可）",
        )

        with gr.Row():
            refresh_btn = gr.Button("更新")
            frame_btn = gr.Button("Extract Frames")
            face_btn = gr.Button("Extract Faces")
            cluster_btn = gr.Button("Cluster Faces")
            analyze_btn = gr.Button("Analyze", variant="primary")
            batch_analyze_btn = gr.Button("選択した動画を連続Analyze", variant="primary")

        result = gr.Textbox(label="ログ", lines=15)
        gallery = gr.Gallery(
            label="人物一覧", columns=5, rows=2, height=300,
            object_fit="contain", preview=True,
        )
        selected_face = gr.State()
        selected_cluster = gr.State()
        selected_face_detail = gr.Textbox(
            label="選択した顔画像",
            value="顔画像を選択してください",
            interactive=False,
        )
        exclude_face_btn = gr.Button("選択した顔をデータベースから除外", variant="stop")
        cluster_actor_name = gr.Textbox(label="Personを出演者名として登録")
        cluster_merge_target = gr.Dropdown(
            label="既存の出演者に統合（任意）",
            choices=[],
        )
        register_cluster_btn = gr.Button("出演者ライブラリへ登録")

        gr.Markdown("## 出演者ライブラリ")

        with gr.Row():
            refresh_actor_btn = gr.Button("出演者更新")
            actor_count = gr.Textbox(label="登録数", value="出演者数：0", interactive=False)

        with gr.Row():
            actor_sort_by = gr.Dropdown(
                label="並び替え基準",
                choices=[("名前", "name"), ("顔画像枚数", "count")],
                value="name",
            )
            actor_sort_order = gr.Radio(
                label="並び順",
                choices=[("昇順", "asc"), ("降順", "desc")],
                value="asc",
            )

        actor_gallery = gr.Gallery(
            label="出演者", columns=5, rows=2, height=300,
            object_fit="contain", preview=True,
        )

        selected_actor = gr.State()
        actor_name = gr.Textbox(label="出演者名")
        actor_detail = gr.Textbox(label="詳細", value="出演者を選択してください", interactive=False)
        actor_faces = gr.Gallery(
            label="顔画像一覧", columns=6, rows=2, height=260,
            object_fit="contain", preview=True,
        )

        with gr.Row():
            rename_actor_btn = gr.Button("名前を変更")
            delete_actor_btn = gr.Button("削除", variant="stop")

        with gr.Row():
            merge_target = gr.Dropdown(label="統合先（類似順）", choices=[])
            merge_result_name = gr.Dropdown(
                label="統合後の出演者名",
                choices=[],
                allow_custom_value=True,
            )
            merge_actor_btn = gr.Button("統合")

        actor_message = gr.Textbox(label="出演者ライブラリの操作結果", interactive=False)

        refresh_btn.click(refresh, inputs=scan_directory, outputs=[video, batch_videos, result])
        video.change(selected, video, result)
        frame_btn.click(extract_frames_with_progress, video, result)
        face_btn.click(extract_faces_with_progress, video, result)
        cluster_btn.click(cluster_and_gallery, video, outputs=[result, gallery])
        analyze_btn.click(analyze_video, video, outputs=[result, gallery])
        batch_analyze_btn.click(analyze_multiple_videos, batch_videos, outputs=[result, gallery])

        gallery.select(
            select_detected_face,
            inputs=video,
            outputs=[
                selected_face,
                selected_cluster,
                cluster_actor_name,
                cluster_merge_target,
                selected_face_detail,
            ],
        )
        exclude_face_btn.click(
            exclude_selected_face,
            inputs=[video, selected_face],
            outputs=[selected_face, gallery, selected_face_detail, result],
        )
        refresh_actor_btn.click(
            refresh_actor_library,
            inputs=[actor_sort_by, actor_sort_order],
            outputs=[actor_gallery, actor_count],
        )
        actor_sort_by.change(
            refresh_actor_library,
            inputs=[actor_sort_by, actor_sort_order],
            outputs=[actor_gallery, actor_count],
        )
        actor_sort_order.change(
            refresh_actor_library,
            inputs=[actor_sort_by, actor_sort_order],
            outputs=[actor_gallery, actor_count],
        )
        register_cluster_btn.click(
            register_selected_cluster,
            inputs=[video, selected_cluster, cluster_actor_name, cluster_merge_target],
            outputs=[actor_gallery, actor_count, actor_message],
        )

        actor_gallery.select(
            select_actor,
            inputs=[actor_sort_by, actor_sort_order],
            outputs=[
                selected_actor,
                actor_name,
                actor_faces,
                actor_detail,
                merge_target,
                merge_result_name,
            ],
        )
        merge_target.change(
            update_merge_result_name,
            inputs=[selected_actor, merge_target],
            outputs=merge_result_name,
        )

        library_outputs = [
            selected_actor, actor_gallery, actor_count, actor_name, actor_faces,
            actor_detail, merge_target, actor_message,
        ]
        rename_actor_btn.click(
            rename_selected_actor,
            inputs=[selected_actor, actor_name, actor_sort_by, actor_sort_order],
            outputs=library_outputs,
        )
        delete_actor_btn.click(
            delete_selected_actor,
            inputs=[selected_actor, actor_sort_by, actor_sort_order],
            outputs=library_outputs,
        )
        merge_actor_btn.click(
            merge_selected_actors,
            inputs=[
                selected_actor,
                merge_target,
                merge_result_name,
                actor_sort_by,
                actor_sort_order,
            ],
            outputs=library_outputs,
        )

        gr.Markdown("## データ初期化")
        initialize_confirmed = gr.Checkbox(
            label="初期化により対象データが完全に削除されることを確認しました"
        )

        with gr.Row():
            initialize_database_btn = gr.Button("データベースを初期化", variant="stop")
            initialize_library_btn = gr.Button("出演者ライブラリを初期化", variant="stop")

        initialize_message = gr.Textbox(label="初期化結果", interactive=False)

        initialize_database_btn.click(
            initialize_database,
            inputs=initialize_confirmed,
            outputs=[initialize_confirmed, initialize_message],
        )
        initialize_library_btn.click(
            initialize_actor_library,
            inputs=initialize_confirmed,
            outputs=[
                initialize_confirmed, selected_actor, actor_gallery, actor_count,
                actor_name, actor_faces, actor_detail, merge_target,
                initialize_message,
            ],
        )

        gr.Markdown("## データベースビューアー")
        refresh_database_btn = gr.Button("データベースを更新")
        database_table = gr.Dropdown(label="テーブル", choices=[])
        database_data = gr.Dataframe(
            label="テーブル内容",
            value=[[""]],
            headers=["データなし"],
            column_count=1,
            interactive=False,
        )
        database_message = gr.Textbox(label="データベース表示", interactive=False)

        refresh_database_btn.click(
            refresh_database_viewer,
            outputs=[database_table, database_data, database_message],
        )
        database_table.change(
            select_database_table,
            inputs=database_table,
            outputs=[database_data, database_message],
        )

    return demo
