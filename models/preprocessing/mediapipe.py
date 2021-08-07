import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def pose_estimation(image, holistic):
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic.process(image)

    image = np.zeros(shape=image.shape, dtype=np.float)
    landmark_subset = landmark_pb2.NormalizedLandmarkList(
        landmark=[
                     results.pose_landmarks.landmark[i] for i in range(17)
                 ] + [
                     results.pose_landmarks.landmark[23],
                     results.pose_landmarks.landmark[24]
                 ]
    )

    pose_connection = [(0, 1),
                       (1, 2),
                       (2, 3),
                       (3, 7),
                       (0, 4),
                       (4, 5),
                       (5, 6),
                       (6, 8),
                       (9, 10),
                       (11, 12),
                       (12, 14),
                       (14, 16),
                       (11, 13),
                       (13, 15),
                       (11, 17),
                       (12, 18),
                       (17, 18)]

    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, landmark_subset, pose_connection)

    return image
