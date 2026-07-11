from pathlib import Path
import shutil
import numpy as np

from config import PERSON_LIBRARY_DIR
from sklearn.metrics.pairwise import cosine_similarity


def get_next_person_id():

    person_dirs = sorted(PERSON_LIBRARY_DIR.glob("Person*"))

    if len(person_dirs) == 0:
        return 1

    numbers = []

    for folder in person_dirs:

        try:
            numbers.append(
                int(folder.name.replace("Person", ""))
            )
        except ValueError:
            pass

    return max(numbers) + 1


def create_person(face_path, embedding_path):

    person_id = get_next_person_id()

    person_name = f"Person{person_id:06d}"

    person_dir = PERSON_LIBRARY_DIR / person_name

    person_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        face_path,
        person_dir / "representative.jpg"
    )

    shutil.copy2(
        embedding_path,
        person_dir / "representative.npy"
    )

    return person_name


def load_all_persons():

    persons = []

    for folder in sorted(PERSON_LIBRARY_DIR.glob("Person*")):

        image = folder / "representative.jpg"

        embedding = folder / "representative.npy"

        if image.exists() and embedding.exists():

            persons.append({

                "name": folder.name,

                "image": image,

                "embedding": np.load(embedding)

            })

    return persons

def find_similar_actor(embedding, threshold=0.65):

    actors = load_all_persons()

    best_actor = None
    best_score = -1

    embedding = embedding.reshape(1, -1)

    for actor in actors:

        score = cosine_similarity(
            embedding,
            actor["embedding"].reshape(1, -1)
        )[0][0]

        if score > best_score:
            best_score = score
            best_actor = actor

    if best_score >= threshold:
        return best_actor

    return None

def update_actor_embedding(actor_name):

    actor_dir = PERSON_LIBRARY_DIR / actor_name

    faces_dir = actor_dir / "faces"

    embeddings = []

    for image in sorted(faces_dir.glob("*.jpg")):

        npy = image.with_suffix(".npy")

        if npy.exists():

            embeddings.append(
                np.load(npy)
            )

    if len(embeddings) == 0:
        return False

    mean_embedding = np.mean(
        embeddings,
        axis=0
    )

    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    np.save(
        actor_dir / "representative.npy",
        mean_embedding
    )

    return True