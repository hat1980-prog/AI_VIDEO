from pathlib import Path

from config import FACES_DIR


def load_gallery(video_name):

    video_name = Path(video_name).stem

    base_dir = FACES_DIR / video_name

    gallery = []

    if not base_dir.exists():
        return gallery

    folders = sorted(base_dir.glob("Person_*"))

    unknown = base_dir / "Unknown"

    if unknown.exists():
        folders.append(unknown)

    for folder in folders:

        for image in sorted(folder.glob("*.jpg")):

            gallery.append(
                (
                    str(image),
                    folder.name
                )
            )

    return gallery