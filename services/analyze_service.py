from services.frame_service import extract_frames
from services.face_service import extract_faces
from services.cluster_service import cluster_faces
from services.gallery_service import load_gallery


def analyze_video(video_name):

    if not video_name:
        return "動画を選択してください", []

    logs = []

    ok, msg = extract_frames(video_name)
    logs.append(msg)

    if not ok:
        return "\n".join(logs), []

    logs.append(extract_faces(video_name))
    logs.append(cluster_faces(video_name))

    gallery = load_gallery(video_name)

    return "\n".join(logs), gallery