from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import cv2
import threading
import numpy as np
import time
import face_recognition

app = FastAPI()

# Shared variables
latest_gender = None
latest_age_group = None
lock = threading.Lock()
roi = None
prev_embedding = None

# Pretrained model paths
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Model Mean Values and Labels
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

drawing = False
roi_start = (0, 0)
roi_end = (0, 0)
roi_defined = False


class PersonResponse(BaseModel):
    gender: str
    age_group: str


def draw_roi(event, x, y, flags, param):
    global drawing, roi_start, roi_end, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)
        roi_defined = True
        print(f"[INFO] ROI set from {roi_start} to {roi_end}")


'''def detect_face_attributes():
    global latest_gender, latest_age_group, roi, roi_start, roi_end, roi_defined, prev_embedding

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", draw_roi)

    # Wait until ROI is drawn
    while not roi_defined:
        ret, frame = cap.read()
        if not ret:
            continue
        temp_frame = frame.copy()
        if drawing:
            cv2.rectangle(temp_frame, roi_start, roi_end, (255, 0, 0), 2)
        cv2.imshow("Draw ROI", temp_frame)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow("Draw ROI")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        x1, y1 = roi_start
        x2, y2 = roi_end
        roi_frame = frame[y1:y2, x1:x2]

        blob = cv2.dnn.blobFromImage(roi_frame, 1.0, (300, 300), [104, 117, 123], True, False)
        faceNet.setInput(blob)
        detections = faceNet.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                fx1 = int(detections[0, 0, i, 3] * roi_frame.shape[1])
                fy1 = int(detections[0, 0, i, 4] * roi_frame.shape[0])
                fx2 = int(detections[0, 0, i, 5] * roi_frame.shape[1])
                fy2 = int(detections[0, 0, i, 6] * roi_frame.shape[0])
                face = roi_frame[fy1:fy2, fx1:fx2]

                # Gender detection
                face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(face_blob)
                gender_preds = genderNet.forward()
                gender = genderList[gender_preds[0].argmax()]

                # Age detection
                ageNet.setInput(face_blob)
                age_preds = ageNet.forward()
                age = ageList[age_preds[0].argmax()]

                with lock:
                    latest_gender = gender
                    latest_age_group = age

                print(f"[INFO] Gender: {gender}, Age: {age}")        

                        

                cv2.rectangle(roi_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

        cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == 27:
            break
        time.sleep(2)
    cap.release()
    cv2.destroyAllWindows()'''

def detect_person_background():
    global latest_gender, latest_age_group, detection_thread_running

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Define ROI centered on screen (e.g., 50% of width and height)
        roi_w, roi_h = int(width * 0.5), int(height * 0.5)
        x1 = (width - roi_w) // 2
        y1 = (height - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        roi = frame[y1:y2, x1:x2]

        # Run face detection in ROI
        blob = cv2.dnn.blobFromImage(roi, 1.0, (300, 300), [104, 117, 123], True, False)
        faceNet.setInput(blob)
        detections = faceNet.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                fx1 = int(detections[0, 0, i, 3] * roi.shape[1])
                fy1 = int(detections[0, 0, i, 4] * roi.shape[0])
                fx2 = int(detections[0, 0, i, 5] * roi.shape[1])
                fy2 = int(detections[0, 0, i, 6] * roi.shape[0])
                face = roi[fy1:fy2, fx1:fx2]

                if face.size == 0:
                    continue

                face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                genderNet.setInput(face_blob)
                genderPreds = genderNet.forward()
                latest_gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(face_blob)
                agePreds = ageNet.forward()
                latest_age_group = ageList[agePreds[0].argmax()]

                print(f"[DETECTED] Gender: {latest_gender}, Age: {latest_age_group}")

        # Draw ROI box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.putText(frame, "Centered ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == 27:
            break

        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()
    detection_thread_running = False



@app.get("/start_system")
def start_system():
    threading.Thread(target=detect_person_background, daemon=True).start()
    return {"message": "System started, camera running in background"}


@app.get("/get_person_details", response_model=PersonResponse)
def get_person_details():
    with lock:
        if latest_gender and latest_age_group:
            return PersonResponse(gender=latest_gender, age_group=latest_age_group)
        else:
            return JSONResponse(status_code=404, content={"message": "No face detected yet"})
