from pathlib import Path

from config import FACES_DIR
from services.video_service import get_video_id


def load_gallery(video_name):

    video_id = get_video_id(video_name)

    if video_id is None:
        return []

    base_dir = FACES_DIR / video_id

    gallery = []

    if not base_dir.exists():
        return gallery

    folders = sorted(base_dir.glob("Actor_*"))
    folders.extend(sorted(base_dir.glob("Person_*")))

    unknown = base_dir / "Unknown"

    if unknown.exists():
        folders.append(unknown)

    unclassified = base_dir / "Unclassified"

    if unclassified.exists():
        folders.append(unclassified)

    for folder in folders:

        for image in sorted(folder.glob("*.jpg")):

            label = folder.name.removeprefix("Actor_")

            if folder.name == "Unclassified":
                label = "未分類"

            gallery.append((str(image), label))

    return gallery
