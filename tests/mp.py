import cv2
import mediapipe as mp
from models.preprocessing.mediapipe import pose_estimation
import numpy as np
from mediapipe.framework.formats import landmark_pb2

mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture()
cap.open("/Users/wayenvan/Desktop/MscProject/template.mp4")

with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2) as holistic:

    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        image = pose_estimation(image, holistic)
        image = cv2.resize(image, (256, 256))
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
