# main.py
import cv2
import threading
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend access (important for browser fetch calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold latest gender and age
latest_gender = "Unknown"
latest_age = "Unknown"

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
           '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

camera_running = False

# --------------------------
# Face Detection Logic
# --------------------------
def detect_face_attributes():
    global latest_gender, latest_age, camera_running

    cap = cv2.VideoCapture(0)
    camera_running = True

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     [104, 117, 123], False, False)
        faceNet.setInput(blob)
        detections = faceNet.forward()

        h, w = frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                  MODEL_MEAN_VALUES, swapRB=False)
                # Gender
                genderNet.setInput(face_blob)
                gender_preds = genderNet.forward()
                latest_gender = genderList[gender_preds[0].argmax()]

                # Age
                ageNet.setInput(face_blob)
                age_preds = ageNet.forward()
                latest_age = ageList[age_preds[0].argmax()]

                print(f"[INFO] Gender: {latest_gender}, Age: {latest_age}")

        time.sleep(1)  # Limit processing frequency

    cap.release()


# --------------------------
# Start Camera API
# --------------------------
@app.get("/start_camera")
def start_camera():
    global camera_running
    if not camera_running:
        thread = threading.Thread(target=detect_face_attributes, daemon=True)
        thread.start()
        return {"message": "Camera started and running in background."}
    else:
        return {"message": "Camera is already running."}


# --------------------------
# Get Person Details API
# --------------------------
class PersonResponse(BaseModel):
    gender: str
    age_group: str

@app.get("/get_person_details", response_model=PersonResponse)
def get_person_details():
    return {
        "gender": latest_gender,
        "age_group": latest_age
    }




