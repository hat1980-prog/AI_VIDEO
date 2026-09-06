from pathlib import Path
from time import perf_counter

import gradio as gr

from config import FRAME_INTERVAL, VIDEOS_DIR
from services.video_service import get_video_list
from services.database_viewer_service import get_database_tables, get_table_data
from services.exclusion_service import exclude_face
from services.person_registration_service import register_cluster_as_actor
from services.classification_service import (
    exclude_from_classification,
    get_face_similarity_candidates,
    get_manual_targets,
    reclassify_face,
)
from services.frame_service import extract_frames
from services.face_service import extract_faces
from services.cluster_service import cluster_faces
from services.gallery_service import load_gallery
from services.analyze_service import analyze_video
from services.actor_service import (
    delete_actor,
    delete_actors,
    get_actor_details,
    get_similar_actor_choices,
    get_actor_video_choices,
    load_actor_library,
    load_actor_list,
    merge_actors,
    merge_actors_into_target,
    reclassify_actors,
    rename_actor,
    reclassify_actor_face,
    reclassify_actor_faces,
    remove_actor_face,
    set_actor_representative_image,
)
from services.person_library_service import get_similarity_threshold
from services.reset_service import reset_actor_library, reset_database
from services.video_data_service import delete_unknown_source_faces, delete_video_analysis_data
from services.cancellation_service import (
    begin_analysis,
    is_analysis_cancelled,
    request_analysis_cancel,
    request_current_video_skip,
)
from services.runtime_log_service import append_runtime_log, clear_runtime_log, get_runtime_log


def refresh(scan_directory, exclude_analyzed, progress=gr.Progress()):
    progress(0.1, desc="動画一覧を取得しています")
    choices = get_video_list(scan_directory, exclude_analyzed)
    progress(1.0, desc="動画一覧を更新しました")
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=[]),
        f"{len(choices)}件の動画を検出しました",
    )


def select_all_batch_videos(scan_directory, exclude_analyzed, progress=gr.Progress()):
    progress(0.1, desc="動画一覧を取得しています")
    choices = get_video_list(scan_directory, exclude_analyzed)
    progress(1.0, desc=f"{len(choices)}件の動画を選択しました")
    return gr.update(choices=choices, value=[value for _, value in choices])


def clear_batch_videos():
    return gr.update(value=[])


def selected(video):
    if not video:
        return "動画を選択してください", []

    return f"選択中：{Path(video).name}", load_gallery(video)


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


def analyze_multiple_videos(videos, frame_interval, progress=gr.Progress()):
    if not videos:
        return "連続Analyzeの対象動画を選択してください", []

    logs = []
    gallery = []
    total = len(videos)
    begin_analysis()
    clear_runtime_log()
    started_at = perf_counter()

    for index, video in enumerate(videos):
        if is_analysis_cancelled():
            logs.append("中断要求を受けたため、連続Analyzeを終了しました")
            break
        video_name = Path(video).name

        def update_progress(value, desc):
            percentage = (index + value) / total
            progress(percentage, desc=f"{index + 1}/{total}: {video_name} - {desc}")

        update_progress(0.0, "解析を開始しています")
        append_runtime_log(f"[連続Analyze] {index + 1}/{total} 開始: {video_name}")
        video_started_at = perf_counter()
        result, video_gallery = analyze_video(
            video, frame_interval, progress=update_progress, reset_cancel=False
        )
        logs.append(f"=== {video_name}（{perf_counter() - video_started_at:.1f}秒）===\n{result}")
        gallery.extend(video_gallery)
        append_runtime_log(f"[連続Analyze] {index + 1}/{total} 終了: {video_name}")

    progress(1.0, desc="連続Analyzeを中断しました" if is_analysis_cancelled() else "選択したすべての動画の解析が完了しました")
    logs.append(f"連続Analyze合計: {perf_counter() - started_at:.1f}秒")
    return "\n\n".join(logs), gallery


def cancel_running_analysis():
    request_analysis_cancel()
    return "中断要求を受け付けました。実行中の処理単位が完了次第、解析を停止します"


def skip_current_video_analysis():
    request_current_video_skip()
    return "現在の動画をスキップします。処理中の段階が終了次第、次の動画のAnalyzeへ進みます"


def extract_frames_with_progress(video, frame_interval, progress=gr.Progress()):
    progress(0.05, desc="フレーム抽出を開始しています")
    ok, message = extract_frames(video, frame_interval)
    progress(1.0, desc="フレーム抽出が完了しました" if ok else "フレーム抽出に失敗しました")
    return message


def extract_faces_with_progress(video, frame_interval, progress=gr.Progress()):
    progress(0.05, desc="顔検出を開始しています")

    def update_progress(value):
        progress(value, desc=f"顔を検出しています（{int(value * 100)}%）")

    message = extract_faces(
        video,
        progress_callback=update_progress,
        frame_interval=frame_interval,
    )
    progress(1.0, desc="顔検出が完了しました")
    return message


def delete_selected_video_data(video, confirmed, progress=gr.Progress()):
    if not confirmed:
        return False, [], "削除するには確認チェックを入れてください"

    progress(0.1, desc="動画ごとの解析データを削除しています")
    message = delete_video_analysis_data(video)
    progress(1.0, desc="動画ごとの解析データを削除しました")
    return False, [], message


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
            gr.update(choices=[], value=None),
            "顔画像を選択してください",
        )

    face_path, label = faces[evt.index]
    folder_name = Path(face_path).parent.name
    selected_cluster = folder_name if folder_name.startswith("Person_") else None
    actor_name = selected_cluster or ""

    similarity_text = ""
    if folder_name.startswith("Person_") or folder_name in {"Unknown", "Unclassified"}:
        candidates = get_face_similarity_candidates(video, face_path)
        if candidates:
            similarity_text = "\n候補一致度: " + " / ".join(
                f"{name} {similarity:.2f}" for name, similarity in candidates
            )
        else:
            similarity_text = "\n候補一致度: 出演者ライブラリのEmbeddingがありません"

    return (
        face_path,
        selected_cluster,
        actor_name,
        gr.update(choices=load_actor_list(), value=None),
        gr.update(choices=get_manual_targets(video), value=None),
        f"選択中：{Path(face_path).name}（{label}）{similarity_text}",
    )


def exclude_selected_face(video, face_path, progress=gr.Progress()):
    progress(0.1, desc="顔画像を除外しています")
    message = exclude_face(video, face_path)
    progress(0.85, desc="顔一覧を更新しています")
    gallery = load_gallery(video)
    progress(1.0, desc="除外処理が完了しました")
    return None, gallery, "顔画像を選択してください", message


def exclude_selected_face_from_classification(video, face_path, progress=gr.Progress()):
    progress(0.1, desc="顔画像を分類から除外しています")
    message = exclude_from_classification(video, face_path)
    progress(0.85, desc="人物一覧を更新しています")
    gallery = load_gallery(video)
    progress(1.0, desc="分類からの除外が完了しました")
    return None, gallery, gr.update(choices=get_manual_targets(video), value=None), "顔画像を選択してください", message


def manually_reclassify_selected_face(video, face_path, target, progress=gr.Progress()):
    progress(0.1, desc="顔画像を手動分類しています")
    message = reclassify_face(video, face_path, target)
    progress(0.85, desc="人物一覧を更新しています")
    gallery = load_gallery(video)
    progress(1.0, desc="手動分類が完了しました")
    return None, gallery, gr.update(choices=get_manual_targets(video), value=None), "顔画像を選択してください", message


def register_selected_cluster(
    video, cluster_name, actor_name, merge_target, sort_by, sort_order, source_video_path, actor_name_filter,
    progress=gr.Progress(),
):
    progress(0.1, desc="出演者ライブラリへ登録しています")
    target_actor_name = merge_target or actor_name
    message = register_cluster_as_actor(video, cluster_name, target_actor_name)
    progress(0.85, desc="出演者ライブラリを更新しています")
    actor_gallery, actor_count = load_actor_library(
        sort_by, sort_order, source_video_path, actor_name_filter
    )
    progress(1.0, desc="出演者登録が完了しました")
    return actor_gallery, actor_count, message


def select_actor(sort_by, sort_order, source_video_path, actor_name_filter, evt: gr.SelectData):
    gallery, _ = load_actor_library(
        sort_by, sort_order, source_video_path, actor_name_filter
    )

    if evt.index is None or evt.index >= len(gallery):
        return (
            None,
            "",
            [],
            "出演者を選択してください",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            [],
            [],
        )

    actor_name = gallery[evt.index][1]
    name, faces, detail = get_actor_details(actor_name, source_video_path)
    choices = get_similar_actor_choices(actor_name)

    return (
        actor_name,
        name,
        faces,
        detail,
        gr.update(choices=choices, value=None),
        gr.update(choices=[], value=None),
        [],
        [],
    )


def select_actor_face(actor_name, source_video_path, selected_face_paths, evt: gr.SelectData):
    _, faces, _ = get_actor_details(actor_name, source_video_path)

    if evt.index is None or evt.index >= len(faces):
        return (
            None,
            "出演者ライブラリの顔画像を選択してください",
            gr.update(choices=[], value=None),
            [],
            [],
        )

    face_path, caption = faces[evt.index]
    targets = [name for name in load_actor_list() if name != actor_name]
    selected = list(selected_face_paths or [])

    if face_path in selected:
        selected.remove(face_path)
    else:
        selected.append(face_path)

    selected_set = set(selected)
    selected_gallery = [(path, label) for path, label in faces if path in selected_set]
    return (
        face_path,
        f"選択中：{caption}（移動対象：{len(selected)}枚）",
        gr.update(choices=targets, value=None),
        selected,
        selected_gallery,
    )


def _actor_face_operation_outputs(
    source_actor_name, sort_by, sort_order, source_video_path, actor_name_filter, message
):
    gallery, count = load_actor_library(
        sort_by, sort_order, source_video_path, actor_name_filter
    )

    if source_actor_name in load_actor_list():
        selected_name, faces, detail = get_actor_details(source_actor_name, source_video_path)
        merge_choices = get_similar_actor_choices(source_actor_name)
        face_targets = [name for name in load_actor_list() if name != source_actor_name]
    else:
        selected_name, faces, detail = None, [], "出演者を選択してください"
        merge_choices, face_targets = [], []

    return (
        selected_name,
        gallery,
        count,
        selected_name or "",
        faces,
        detail,
        gr.update(choices=merge_choices, value=None),
        None,
        "出演者ライブラリの顔画像を選択してください",
        gr.update(choices=face_targets, value=None),
        [],
        [],
        message,
    )


def remove_selected_actor_face(
    actor_name, face_path, sort_by, sort_order, source_video_path, actor_name_filter,
    progress=gr.Progress(),
):
    progress(0.1, desc="出演者ライブラリから顔画像を除外しています")
    message = remove_actor_face(actor_name, face_path)
    progress(0.85, desc="出演者ライブラリを更新しています")
    outputs = _actor_face_operation_outputs(
        actor_name, sort_by, sort_order, source_video_path, actor_name_filter, message
    )
    progress(1.0, desc="顔画像の除外が完了しました")
    return outputs


def reclassify_selected_actor_face(
    actor_name, face_path, target_actor_name, sort_by, sort_order, source_video_path, actor_name_filter,
    progress=gr.Progress(),
):
    progress(0.1, desc="出演者ライブラリの顔画像を再分類しています")
    message = reclassify_actor_face(actor_name, face_path, target_actor_name)
    progress(0.85, desc="出演者ライブラリを更新しています")
    outputs = _actor_face_operation_outputs(
        actor_name, sort_by, sort_order, source_video_path, actor_name_filter, message
    )
    progress(1.0, desc="顔画像の再分類が完了しました")
    return outputs


def reclassify_selected_actor_faces(
    actor_name, face_paths, target_actor_name, sort_by, sort_order, source_video_path,
    actor_name_filter, progress=gr.Progress(),
):
    progress(0.1, desc="複数の顔画像を別の出演者へ移動しています")
    message = reclassify_actor_faces(actor_name, face_paths, target_actor_name)
    progress(0.85, desc="出演者ライブラリを更新しています")
    outputs = _actor_face_operation_outputs(
        actor_name, sort_by, sort_order, source_video_path, actor_name_filter, message
    )
    progress(1.0, desc="複数画像の移動が完了しました")
    return (*outputs, "")


def set_selected_actor_thumbnail(
    actor_name, face_path, sort_by, sort_order, source_video_path, actor_name_filter,
    progress=gr.Progress(),
):
    progress(0.1, desc="出演者サムネイルを変更しています")
    message = set_actor_representative_image(actor_name, face_path)
    progress(0.85, desc="出演者ライブラリを更新しています")
    outputs = _actor_face_operation_outputs(
        actor_name, sort_by, sort_order, source_video_path, actor_name_filter, message
    )
    progress(1.0, desc="出演者サムネイルを変更しました")
    return outputs


def _refresh_actor_filter_state(sort_by, sort_order, source_video_path, actor_name_filter, priority):
    if priority == "actor" and actor_name_filter:
        video_choices = [("すべての元動画", None), *get_actor_video_choices(actor_name_filter)]
        valid_video_paths = {value for _, value in video_choices}
        source_video_path = source_video_path if source_video_path in valid_video_paths else None
    else:
        video_choices = [("すべての元動画", None), *get_actor_video_choices()]
        valid_video_paths = {value for _, value in video_choices}
        source_video_path = source_video_path if source_video_path in valid_video_paths else None

    actor_choices = [
        ("すべての出演者", None),
        *[(name, name) for name in load_actor_list(source_video_path=source_video_path)],
    ]
    valid_actor_names = {value for _, value in actor_choices}
    actor_name_filter = actor_name_filter if actor_name_filter in valid_actor_names else None

    if actor_name_filter:
        video_choices = [("すべての元動画", None), *get_actor_video_choices(actor_name_filter)]
        valid_video_paths = {value for _, value in video_choices}
        source_video_path = source_video_path if source_video_path in valid_video_paths else None

    gallery, count = load_actor_library(
        sort_by, sort_order, source_video_path, actor_name_filter
    )

    return (
        gallery,
        count,
        gr.update(choices=video_choices, value=source_video_path),
        gr.update(
            choices=actor_choices,
            value=actor_name_filter,
        ),
        None,
        "",
        [],
        "出演者を選択してください",
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        None,
        "出演者ライブラリの顔画像を選択してください",
        gr.update(choices=[], value=None),
        [],
        [],
    )


def refresh_actor_library(sort_by, sort_order, source_video_path, actor_name_filter, progress=gr.Progress()):
    progress(0.1, desc="出演者ライブラリを更新しています")
    outputs = _refresh_actor_filter_state(
        sort_by, sort_order, source_video_path, actor_name_filter, "video"
    )
    progress(1.0, desc="出演者ライブラリを更新しました")
    return outputs


def refresh_actor_library_from_video(sort_by, sort_order, source_video_path, actor_name_filter, progress=gr.Progress()):
    progress(0.1, desc="元動画に紐づく出演者を絞り込んでいます")
    outputs = _refresh_actor_filter_state(
        sort_by, sort_order, source_video_path, actor_name_filter, "video"
    )
    progress(1.0, desc="出演者一覧を更新しました")
    return outputs


def refresh_actor_library_from_actor(sort_by, sort_order, source_video_path, actor_name_filter, progress=gr.Progress()):
    progress(0.1, desc="出演者に紐づく元動画を絞り込んでいます")
    outputs = _refresh_actor_filter_state(
        sort_by, sort_order, source_video_path, actor_name_filter, "actor"
    )
    progress(1.0, desc="出演者一覧を更新しました")
    return outputs


def rename_selected_actor(
    actor_name, new_name, sort_by, sort_order, source_video_path, actor_name_filter,
    progress=gr.Progress(),
):
    progress(0.1, desc="出演者名を変更しています")
    selected_name, message = rename_actor(actor_name, new_name)
    gallery, count = load_actor_library(
        sort_by, sort_order, source_video_path, actor_name_filter
    )
    name, faces, detail = get_actor_details(selected_name, source_video_path)
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


def delete_selected_actor(
    actor_name, sort_by, sort_order, source_video_path, actor_name_filter, progress=gr.Progress()
):
    progress(0.1, desc="出演者を削除しています")
    message = delete_actor(actor_name)
    gallery, count = load_actor_library(
        sort_by, sort_order, source_video_path, actor_name_filter
    )
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


def refresh_bulk_delete_actor_choices(
    sort_by="name", sort_order="asc", source_video_path=None, actor_name_filter=None,
    selected_names=None,
):
    """Keep bulk-delete candidates aligned with the library filters.

    Unlike bulk merge, an actor-name filter is a true restriction here: it is
    safer for a destructive operation to offer only what the user is viewing.
    """
    choices = load_actor_list(sort_by, sort_order, source_video_path)
    if actor_name_filter:
        choices = [name for name in choices if name == actor_name_filter]
    selected = [name for name in (selected_names or []) if name in choices]
    return gr.update(choices=choices, value=selected)


def refresh_bulk_merge_actor_choices(
    sort_by="name", sort_order="asc", source_video_path=None, actor_name_filter=None,
):
    # The selected actor is a reference only for similarity sorting.  Applying
    # it as a filter here leaves just that one actor and prevents selecting
    # multiple Person entries for bulk merging.
    similarity_reference = actor_name_filter if sort_by == "similarity" else None
    choices = load_actor_list(sort_by, sort_order, source_video_path, similarity_reference)
    return gr.update(choices=choices, value=[]), gr.update(choices=choices, value=None)


def refresh_all_bulk_actor_choices(
    sort_by="name", sort_order="asc", source_video_path=None, actor_name_filter=None,
):
    """Refresh every destructive/bulk candidate list from one actor-library snapshot."""
    delete_choices = refresh_bulk_delete_actor_choices(
        sort_by, sort_order, source_video_path, actor_name_filter
    )
    merge_sources, merge_target = refresh_bulk_merge_actor_choices(
        sort_by, sort_order, source_video_path, actor_name_filter
    )
    return delete_choices, merge_sources, merge_target


def delete_selected_actors(
    actor_names, confirmed, sort_by, sort_order, source_video_path, actor_name_filter,
    progress=gr.Progress(),
):
    if not confirmed:
        gallery, count = load_actor_library(
            sort_by, sort_order, source_video_path, actor_name_filter
        )
        return (
            False,
            refresh_bulk_delete_actor_choices(
                sort_by, sort_order, source_video_path, actor_name_filter, actor_names
            ),
            gallery,
            count,
            "一括削除するには確認チェックを入れてください",
        )

    progress(0.1, desc="選択した出演者を削除しています")
    _, message = delete_actors(actor_names)
    progress(0.85, desc="出演者ライブラリを更新しています")
    gallery, count = load_actor_library(
        sort_by, sort_order, source_video_path, actor_name_filter
    )
    progress(1.0, desc="出演者の一括削除が完了しました")
    return (
        False,
        refresh_bulk_delete_actor_choices(
            sort_by, sort_order, source_video_path, actor_name_filter
        ),
        gallery,
        count,
        message,
    )


def merge_selected_actors_into_target(
    source_names, target_name, confirmed, sort_by, sort_order, source_video_path, actor_name_filter,
    progress=gr.Progress(),
):
    if not confirmed:
        gallery, count = load_actor_library(sort_by, sort_order, source_video_path, actor_name_filter)
        sources, target = refresh_bulk_merge_actor_choices(
            sort_by, sort_order, source_video_path, actor_name_filter
        )
        return False, sources, target, gallery, count, "一括統合するには確認チェックを入れてください"

    progress(0.1, desc="選択した出演者を統合しています")
    _, message = merge_actors_into_target(source_names, target_name)
    progress(0.85, desc="出演者ライブラリを更新しています")
    gallery, count = load_actor_library(sort_by, sort_order, source_video_path, actor_name_filter)
    sources, target = refresh_bulk_merge_actor_choices(
        sort_by, sort_order, source_video_path, actor_name_filter
    )
    progress(1.0, desc="複数出演者の統合が完了しました")
    return False, sources, target, gallery, count, message


def update_merge_result_name(source_name, target_name):
    choices = [name for name in (source_name, target_name) if name]
    return gr.update(choices=choices, value=target_name or None)


def merge_selected_actors(
    source_name, target_name, result_name, sort_by, sort_order, source_video_path, actor_name_filter,
    progress=gr.Progress(),
):
    progress(0.05, desc="出演者を統合しています")
    message = merge_actors(source_name, target_name, result_name)
    progress(0.85, desc="出演者ライブラリを更新しています")
    gallery, count = load_actor_library(
        sort_by, sort_order, source_video_path, actor_name_filter
    )
    progress(1.0, desc="出演者の統合が完了しました")
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


def bulk_reclassify_actor_library(
    confirmed, sort_by, sort_order, source_video_path, actor_name_filter, progress=gr.Progress(),
):
    if not confirmed:
        return (
            None, *load_actor_library(sort_by, sort_order, source_video_path, actor_name_filter),
            "", [], "出演者を選択してください", gr.update(choices=[], value=None),
            "一括再分類を実行するには確認チェックを入れてください",
        )

    begin_analysis()
    threshold = get_similarity_threshold()
    progress(0.02, desc=f"出演者を照合しています（しきい値: {threshold:.2f}）")

    def update_progress(phase, current, total):
        ratio = current / total if total else 1.0
        if phase == "照合":
            progress(0.05 + ratio * 0.45, desc=f"出演者を照合しています（{current}/{total}）")
        else:
            progress(0.50 + ratio * 0.42, desc=f"候補を統合しています（{current}/{total}）")

    merged, cancelled = reclassify_actors(
        threshold,
        progress_callback=update_progress,
        should_cancel=is_analysis_cancelled,
    )
    progress(0.95, desc="出演者ライブラリを更新しています")
    gallery, count = load_actor_library(sort_by, sort_order, source_video_path, actor_name_filter)
    if cancelled:
        message = f"一括再分類を中断しました（統合済み：{len(merged)}人）"
    else:
        message = f"一括再分類が完了しました（しきい値: {threshold:.2f}、統合：{len(merged)}人）"
    progress(1.0, desc="一括再分類を中断しました" if cancelled else "一括再分類が完了しました")
    return (
        None, gallery, count, "", [], "出演者を選択してください",
        gr.update(choices=[], value=None), message,
    )


def initialize_database(confirmed, progress=gr.Progress()):
    if not confirmed:
        return False, "初期化するには確認チェックを入れてください"

    progress(0.1, desc="データベースを初期化しています")
    message = reset_database()
    progress(1.0, desc="データベースの初期化が完了しました")
    return False, message


def delete_unknown_source_faces_from_database(confirmed, progress=gr.Progress()):
    if not confirmed:
        return False, "削除するには確認チェックを入れてください"

    progress(0.1, desc="元動画情報がない顔画像を検索しています")
    message = delete_unknown_source_faces()
    progress(1.0, desc="元動画情報がない顔画像を削除しました")
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
        video = gr.Dropdown(
            choices=get_video_list(VIDEOS_DIR),
            label="顔写真一覧の元動画（フィルター）",
        )
        batch_videos = gr.CheckboxGroup(
            choices=get_video_list(VIDEOS_DIR),
            label="連続Analyze対象（複数選択可）",
        )
        exclude_analyzed_videos = gr.Checkbox(
            label="解析済み動画を除外",
            info="顔画像またはEmbeddingの保存済み動画を一覧から除外します",
        )
        with gr.Row():
            select_all_batch_btn = gr.Button("解析対象を一括選択")
            clear_batch_btn = gr.Button("解析対象を一括選択解除")
        frame_interval = gr.Number(
            label="フレーム抽出間隔（秒）",
            value=FRAME_INTERVAL,
            minimum=1,
            precision=0,
            info="短いほど検出精度が上がり、解析時間と保存容量も増えます",
        )

        with gr.Row():
            refresh_btn = gr.Button("更新")
            frame_btn = gr.Button("Extract Frames")
            face_btn = gr.Button("Extract Faces")
            cluster_btn = gr.Button("Cluster Faces")
            analyze_btn = gr.Button("Analyze", variant="primary")
            batch_analyze_btn = gr.Button("選択した動画を連続Analyze", variant="primary")
            skip_video_btn = gr.Button("現在の動画をスキップ")
            cancel_analyze_btn = gr.Button("Analyzeを中断", variant="stop")

        analysis_cancel_status = gr.Textbox(
            label="Analyze中断状態",
            value="",
            interactive=False,
        )
        runtime_detail_log = gr.Textbox(
            label="実行中詳細（自動更新）",
            value="実行待機中",
            lines=10,
            interactive=False,
        )
        runtime_log_timer = gr.Timer(1.0)

        with gr.Row():
            delete_video_data_confirm = gr.Checkbox(
                label="選択中の元動画の解析データを削除することを確認しました"
            )
            delete_video_data_btn = gr.Button(
                "動画解析データを削除（出演者学習データは保持）",
                variant="stop",
            )

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
        classification_target = gr.Dropdown(label="手動分類先", choices=[])
        with gr.Row():
            exclude_classification_btn = gr.Button("選択した顔を分類から除外", variant="stop")
            reclassify_face_btn = gr.Button("選択した顔を手動で再分類")
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
                choices=[
                    ("名前", "name"),
                    ("顔画像枚数", "count"),
                    ("出演作品数", "works"),
                    ("類似度（高い順）", "similarity"),
                ],
                value="name",
            )
            actor_sort_order = gr.Radio(
                label="並び順",
                choices=[("昇順", "asc"), ("降順", "desc")],
                value="asc",
            )
            actor_video_filter = gr.Dropdown(
                label="元動画で絞り込み",
                choices=[("すべての元動画", None), *get_actor_video_choices()],
                value=None,
            )
            actor_name_filter = gr.Dropdown(
                label="出演者で絞り込み",
                choices=[("すべての出演者", None), *[(name, name) for name in load_actor_list()]],
                value=None,
            )

        with gr.Row():
            bulk_delete_actor_names = gr.Dropdown(
                label="一括削除する出演者（複数選択可）",
                choices=load_actor_list(),
                multiselect=True,
            )
            refresh_bulk_delete_choices_btn = gr.Button("削除対象候補を更新")
        bulk_delete_confirm = gr.Checkbox(
            label="選択した出演者と登録顔画像を削除することを確認しました"
        )
        bulk_delete_actor_btn = gr.Button("選択した出演者を一括削除", variant="stop")

        with gr.Row():
            bulk_merge_source_names = gr.Dropdown(
                label="一括統合する未命名出演者（複数選択可）",
                choices=load_actor_list(),
                multiselect=True,
            )
            bulk_merge_target_name = gr.Dropdown(
                label="一括統合先の出演者",
                choices=load_actor_list(),
            )
            refresh_bulk_merge_choices_btn = gr.Button("統合候補を更新")
        bulk_merge_confirm = gr.Checkbox(
            label="選択した出演者を統合先へまとめて移動することを確認しました"
        )
        bulk_merge_actor_btn = gr.Button("選択した出演者を一括統合")

        actor_gallery = gr.Gallery(
            label="出演者", columns=5, rows=2, height=300,
            object_fit="contain", preview=True,
        )

        selected_actor = gr.State()
        actor_name = gr.Textbox(label="出演者名")
        actor_detail = gr.Textbox(label="詳細", value="出演者を選択してください", interactive=False)
        actor_faces = gr.Gallery(
            label="顔画像一覧（クリックで移動対象に追加・解除）", columns=6, rows=2, height=260,
            object_fit="contain", preview=True,
        )
        selected_actor_faces = gr.State([])
        selected_actor_faces_gallery = gr.Gallery(
            label="移動対象の顔画像", columns=6, rows=1, height=160,
            object_fit="contain", preview=True,
        )
        batch_actor_target = gr.Textbox(
            label="複数画像の移動先出演者名（新規または既存）",
        )
        move_actor_faces_btn = gr.Button("選択した複数画像を別の出演者へ移動")
        selected_actor_face = gr.State()
        actor_face_detail = gr.Textbox(
            label="選択した出演者の顔画像",
            value="出演者ライブラリの顔画像を選択してください",
            interactive=False,
        )
        actor_face_target = gr.Dropdown(
            label="顔画像の再分類先（新規名を入力可）",
            choices=[],
            allow_custom_value=True,
        )
        with gr.Row():
            remove_actor_face_btn = gr.Button("選択した顔をライブラリから除外", variant="stop")
            reclassify_actor_face_btn = gr.Button("選択した顔を別の出演者へ再分類")
            set_actor_thumbnail_btn = gr.Button("選択した顔を出演者サムネイルに設定")

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

        bulk_reclassify_confirm = gr.Checkbox(
            label="現在の自動照合しきい値で、Person形式の出演者を一括統合することを確認しました"
        )
        bulk_reclassify_btn = gr.Button("出演者ライブラリを一括再分類", variant="secondary")

        actor_message = gr.Textbox(label="出演者ライブラリの操作結果", interactive=False)

        refresh_btn.click(
            refresh,
            inputs=[scan_directory, exclude_analyzed_videos],
            outputs=[video, batch_videos, result],
        )
        select_all_batch_btn.click(
            select_all_batch_videos,
            inputs=[scan_directory, exclude_analyzed_videos],
            outputs=batch_videos,
        )
        clear_batch_btn.click(clear_batch_videos, outputs=batch_videos)
        video.change(selected, video, outputs=[result, gallery])
        frame_btn.click(extract_frames_with_progress, [video, frame_interval], result)
        face_btn.click(extract_faces_with_progress, [video, frame_interval], result)
        delete_video_data_btn.click(
            delete_selected_video_data,
            inputs=[video, delete_video_data_confirm],
            outputs=[delete_video_data_confirm, gallery, result],
        )
        cluster_btn.click(cluster_and_gallery, video, outputs=[result, gallery])
        analyze_btn.click(analyze_video, [video, frame_interval], outputs=[result, gallery])
        batch_analyze_btn.click(
            analyze_multiple_videos,
            [batch_videos, frame_interval],
            outputs=[result, gallery],
        )
        cancel_analyze_btn.click(
            cancel_running_analysis,
            outputs=analysis_cancel_status,
            queue=False,
        )
        skip_video_btn.click(
            skip_current_video_analysis,
            outputs=analysis_cancel_status,
            queue=False,
        )
        runtime_log_timer.tick(get_runtime_log, outputs=runtime_detail_log)

        gallery.select(
            select_detected_face,
            inputs=video,
            outputs=[
                selected_face,
                selected_cluster,
                cluster_actor_name,
                cluster_merge_target,
                classification_target,
                selected_face_detail,
            ],
        )
        exclude_face_btn.click(
            exclude_selected_face,
            inputs=[video, selected_face],
            outputs=[selected_face, gallery, selected_face_detail, result],
        )
        exclude_classification_btn.click(
            exclude_selected_face_from_classification,
            inputs=[video, selected_face],
            outputs=[selected_face, gallery, classification_target, selected_face_detail, result],
        )
        reclassify_face_btn.click(
            manually_reclassify_selected_face,
            inputs=[video, selected_face, classification_target],
            outputs=[selected_face, gallery, classification_target, selected_face_detail, result],
        )
        actor_filter_outputs = [
            actor_gallery,
            actor_count,
            actor_video_filter,
            actor_name_filter,
            selected_actor,
            actor_name,
            actor_faces,
            actor_detail,
            merge_target,
            merge_result_name,
            selected_actor_face,
            actor_face_detail,
            actor_face_target,
            selected_actor_faces,
            selected_actor_faces_gallery,
        ]

        refresh_library_event = refresh_actor_btn.click(
            refresh_actor_library,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=actor_filter_outputs,
        )
        refresh_library_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        sort_by_event = actor_sort_by.change(
            refresh_actor_library,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=actor_filter_outputs,
        )
        sort_by_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        sort_order_event = actor_sort_order.change(
            refresh_actor_library,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=actor_filter_outputs,
        )
        sort_order_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        video_filter_event = actor_video_filter.change(
            refresh_actor_library_from_video,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=actor_filter_outputs,
        )
        video_filter_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        actor_filter_event = actor_name_filter.change(
            refresh_actor_library_from_actor,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=actor_filter_outputs,
        )
        actor_filter_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        register_actor_event = register_cluster_btn.click(
            register_selected_cluster,
            inputs=[
                video,
                selected_cluster,
                cluster_actor_name,
                cluster_merge_target,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=[actor_gallery, actor_count, actor_message],
        )
        register_actor_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )

        actor_gallery.select(
            select_actor,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[
                selected_actor,
                actor_name,
                actor_faces,
                actor_detail,
                merge_target,
                merge_result_name,
                selected_actor_faces,
                selected_actor_faces_gallery,
            ],
        )
        actor_faces.select(
            select_actor_face,
            inputs=[selected_actor, actor_video_filter, selected_actor_faces],
            outputs=[
                selected_actor_face,
                actor_face_detail,
                actor_face_target,
                selected_actor_faces,
                selected_actor_faces_gallery,
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
        rename_actor_event = rename_actor_btn.click(
            rename_selected_actor,
            inputs=[
                selected_actor,
                actor_name,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=library_outputs,
        )
        rename_actor_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        delete_actor_event = delete_actor_btn.click(
            delete_selected_actor,
            inputs=[
                selected_actor,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=library_outputs,
        )
        delete_actor_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        refresh_bulk_delete_choices_btn.click(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        bulk_delete_event = bulk_delete_actor_btn.click(
            delete_selected_actors,
            inputs=[
                bulk_delete_actor_names,
                bulk_delete_confirm,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=[
                bulk_delete_confirm,
                bulk_delete_actor_names,
                actor_gallery,
                actor_count,
                actor_message,
            ],
        )
        refresh_bulk_merge_choices_btn.click(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        bulk_delete_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        bulk_merge_event = bulk_merge_actor_btn.click(
            merge_selected_actors_into_target,
            inputs=[
                bulk_merge_source_names,
                bulk_merge_target_name,
                bulk_merge_confirm,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=[
                bulk_merge_confirm,
                bulk_merge_source_names,
                bulk_merge_target_name,
                actor_gallery,
                actor_count,
                actor_message,
            ],
        )
        bulk_merge_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        merge_actor_event = merge_actor_btn.click(
            merge_selected_actors,
            inputs=[
                selected_actor,
                merge_target,
                merge_result_name,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=library_outputs,
        )
        merge_actor_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        bulk_reclassify_event = bulk_reclassify_btn.click(
            bulk_reclassify_actor_library,
            inputs=[
                bulk_reclassify_confirm,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=library_outputs,
        )
        bulk_reclassify_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        actor_face_operation_outputs = [
            selected_actor,
            actor_gallery,
            actor_count,
            actor_name,
            actor_faces,
            actor_detail,
            merge_target,
            selected_actor_face,
            actor_face_detail,
            actor_face_target,
            selected_actor_faces,
            selected_actor_faces_gallery,
            actor_message,
        ]
        remove_actor_face_event = remove_actor_face_btn.click(
            remove_selected_actor_face,
            inputs=[
                selected_actor,
                selected_actor_face,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=actor_face_operation_outputs,
        )
        remove_actor_face_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        reclassify_actor_face_event = reclassify_actor_face_btn.click(
            reclassify_selected_actor_face,
            inputs=[
                selected_actor,
                selected_actor_face,
                actor_face_target,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=actor_face_operation_outputs,
        )
        reclassify_actor_face_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )
        set_actor_thumbnail_btn.click(
            set_selected_actor_thumbnail,
            inputs=[
                selected_actor,
                selected_actor_face,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=actor_face_operation_outputs,
        )
        move_actor_faces_event = move_actor_faces_btn.click(
            reclassify_selected_actor_faces,
            inputs=[
                selected_actor,
                selected_actor_faces,
                batch_actor_target,
                actor_sort_by,
                actor_sort_order,
                actor_video_filter,
                actor_name_filter,
            ],
            outputs=[*actor_face_operation_outputs, batch_actor_target],
        )
        move_actor_faces_event.then(
            refresh_all_bulk_actor_choices,
            inputs=[actor_sort_by, actor_sort_order, actor_video_filter, actor_name_filter],
            outputs=[bulk_delete_actor_names, bulk_merge_source_names, bulk_merge_target_name],
        )

        gr.Markdown("## データ初期化")
        initialize_confirmed = gr.Checkbox(
            label="初期化により対象データが完全に削除されることを確認しました"
        )

        with gr.Row():
            initialize_database_btn = gr.Button("データベースを初期化", variant="stop")
            initialize_library_btn = gr.Button("出演者ライブラリを初期化", variant="stop")
            delete_unknown_source_faces_btn = gr.Button(
                "元動画情報なしの顔画像を一括削除", variant="stop"
            )

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
        delete_unknown_source_faces_btn.click(
            delete_unknown_source_faces_from_database,
            inputs=initialize_confirmed,
            outputs=[initialize_confirmed, initialize_message],
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
