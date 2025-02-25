


import cv2

class FaceEyeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
        self.left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
        self.right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")

    def detect_faces(self, gray_frame):
        """Detects faces in a grayscale image."""
        return self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    def detect_eyes(self, gray_face_region, side="left"):
        """Detects left or right eye in a grayscale face region."""
        if side == "left":
            return self.left_eye_cascade.detectMultiScale(gray_face_region)
        else:
            return self.right_eye_cascade.detectMultiScale(gray_face_region)
