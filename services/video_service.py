import json
from hashlib import sha1
from pathlib import Path

from config import EMBEDDINGS_DIR, FACES_DIR, VIDEO_IDENTITY_INDEX_PATH, VIDEOS_DIR
from services.source_video_service import normalize_renamed_video_identity


VIDEO_PATTERNS = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.mpg", "*.mpeg", "*.wmv", "*.ts")


def _has_analysis_data(video_path):
    video_id = _get_video_id(video_path)
    if video_id is None:
        return False

    # Frame/Embedding folders can be created even if no usable face was found.
    # Only a saved root face image counts as a completed face analysis.
    face_dir = FACES_DIR / video_id
    return face_dir.is_dir() and any(face_dir.glob("*.jpg"))


def _get_video_id_from_path(video_path):
    digest = sha1(str(video_path).encode("utf-8")).hexdigest()[:10]
    return f"{video_path.stem}_{digest}"


def _get_file_identity(video_path):
    try:
        stat = video_path.stat()
    except OSError:
        return None

    # st_ino is stable across a rename on NTFS. Size and modified time prevent
    # stale results from being reused after a file has been replaced in place.
    if not stat.st_ino:
        return None
    return f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"


def _load_video_identity_index():
    if not VIDEO_IDENTITY_INDEX_PATH.exists():
        return {}

    try:
        data = json.loads(VIDEO_IDENTITY_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_video_identity_index(index):
    VIDEO_IDENTITY_INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _get_video_id(video_path):
    video_path = Path(video_path).resolve()
    identity = _get_file_identity(video_path)
    index = _load_video_identity_index()

    if identity and isinstance(index.get(identity), dict):
        record = index[identity]
        video_id = record.get("video_id")
        if video_id:
            # Retain the latest path/name for diagnostics without changing the
            # stable ID used by frames, faces and embeddings.  When the path
            # changed, also migrate source-video labels that belong to this
            # exact filesystem identity.
            if record.get("path") != str(video_path):
                previous_path = record.get("path")
                record["path"] = str(video_path)
                record["name"] = video_path.name
                _save_video_identity_index(index)
                normalize_renamed_video_identity(
                    video_path, video_id, previous_path
                )
            return video_id

    video_id = _get_video_id_from_path(video_path)
    if identity:
        index[identity] = {
            "video_id": video_id,
            "path": str(video_path),
            "name": video_path.name,
        }
        _save_video_identity_index(index)
    return video_id


def _has_stale_source_metadata(video_id, video_path):
    """Whether a completed analysis still displays an old source label."""
    metadata_path = FACES_DIR / video_id / "faces_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_path = str(video_path)
    expected_name = video_path.name
    return any(
        isinstance(info, dict)
        and (
            info.get("video_path") != expected_path
            or info.get("video_name") != expected_name
        )
        for info in metadata.values()
    )


def update_video_file_info(scan_directory=None, progress_callback=None):
    """Refresh stable file identities and follow detected renames/moves.

    This is deliberately a batch operation: the identity index is loaded and
    saved once, even when a scan folder contains thousands of video files.
    """
    root = Path(scan_directory or VIDEOS_DIR).expanduser()
    if not root.is_dir():
        return 0, 0, 0

    root = root.resolve()
    videos = []
    for pattern in VIDEO_PATTERNS:
        videos.extend(root.rglob(pattern))
    videos.sort()

    index = _load_video_identity_index()
    index_changed = False
    renamed_or_moved = 0
    metadata_updates = 0
    total = len(videos)

    for position, video_path in enumerate(videos, start=1):
        video_path = video_path.resolve()
        identity = _get_file_identity(video_path)
        if not identity:
            continue

        record = index.get(identity)
        if isinstance(record, dict) and record.get("video_id"):
            video_id = record["video_id"]
            if record.get("path") != str(video_path):
                previous_path = record.get("path")
                record["path"] = str(video_path)
                record["name"] = video_path.name
                actor_faces, extracted_faces = normalize_renamed_video_identity(
                    video_path, video_id, previous_path
                )
                renamed_or_moved += 1
                metadata_updates += actor_faces + extracted_faces
                index_changed = True
            elif _has_stale_source_metadata(video_id, video_path):
                # Repair source labels when an older run already updated the
                # identity index but did not propagate that rename to faces.
                actor_faces, extracted_faces = normalize_renamed_video_identity(
                    video_path, video_id
                )
                renamed_or_moved += 1
                metadata_updates += actor_faces + extracted_faces
        else:
            index[identity] = {
                "video_id": _get_video_id_from_path(video_path),
                "path": str(video_path),
                "name": video_path.name,
            }
            index_changed = True

        if progress_callback:
            progress_callback(position, total)

    if index_changed:
        _save_video_identity_index(index)

    return total, renamed_or_moved, metadata_updates


def get_video_list(scan_directory=None, exclude_analyzed=False):
    root = Path(scan_directory or VIDEOS_DIR).expanduser()

    if not root.is_dir():
        return []

    root = root.resolve()
    videos = []

    for pattern in VIDEO_PATTERNS:
        videos.extend(root.rglob(pattern))

    videos.sort()

    if exclude_analyzed:
        videos = [video for video in videos if not _has_analysis_data(video)]

    return [
        (str(video.relative_to(root)), str(video.resolve()))
        for video in videos
    ]


def get_video_path(video):
    if not video:
        return None

    video_path = Path(video).expanduser()

    if video_path.is_file():
        return video_path.resolve()

    default_path = VIDEOS_DIR / video_path

    if default_path.is_file():
        return default_path.resolve()

    return None


def get_video_id(video):
    video_path = get_video_path(video)

    if video_path is None:
        return None

    return _get_video_id(video_path)
