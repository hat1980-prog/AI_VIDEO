from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis()
app.prepare(ctx_id=0)

img = cv2.imread("frames/000001.jpg")

faces = app.get(img)

print("faces:", len(faces))