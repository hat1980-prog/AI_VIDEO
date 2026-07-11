from config import ACTORS_DIR


def load_actor_gallery():

    gallery = []

    actor_dirs = sorted(ACTORS_DIR.iterdir())

    for actor_dir in actor_dirs:

        if not actor_dir.is_dir():
            continue

        thumbnail = actor_dir / "thumbnail.jpg"

        if thumbnail.exists():

            gallery.append(
                (
                    str(thumbnail),
                    actor_dir.name
                )
            )

    return gallery

from config import ACTORS_DIR


def load_actor_list():

    actors = []

    actor_dirs = sorted(ACTORS_DIR.iterdir())

    for actor_dir in actor_dirs:

        if actor_dir.is_dir():
            actors.append(actor_dir.name)

    return actors