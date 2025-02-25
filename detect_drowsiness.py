import cv2
import time
from datetime import datetime
from threading import Thread
from models.drowsiness_model import DrowsinessModel
from detection.face_eye_detection import FaceEyeDetector
from sounds.sound_manager import SoundManager

# Initialize Modules
detection = FaceEyeDetector()
drowsiness_model = DrowsinessModel()
sound_manager = SoundManager()

# Capture Video
cap = cv2.VideoCapture(0)
count = 0
alarm_on = False
welcome_played = False

while True:
    _, frame = cap.read()
    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detection.detect_faces(gray)

    # Draw UI
    cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"{current_time}", (400, 20), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Drowsiness Detection System", (10, 20), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        if not welcome_played:
            Thread(target=sound_manager.play_message, args=("data/welcome.mp3", "Hi! I'll be watching over you while driving!")).start()
            welcome_played = True

        left_eyes = detection.detect_eyes(roi_gray, "left")
        right_eyes = detection.detect_eyes(roi_gray, "right")

        status1, status2 = 1, 1  # Default: Eyes Open

        for (x1, y1, w1, h1) in left_eyes:
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            status1 = drowsiness_model.predict_eye_state(eye1)
            break

        for (x2, y2, w2, h2) in right_eyes:
            eye2 = roi_color[y2:y2+h2, x2:x2+w2]
            eye2 = cv2.resize(eye2, (145, 145))
            status2 = drowsiness_model.predict_eye_state(eye2)
            break

        # If both eyes are closed
        if status1 == 2 and status2 == 2:
            count += 1
            cv2.putText(frame, f"Eyes Closed, Count: {count}", (10, 48), cv2.FONT_ITALIC, 0.9, (0, 0, 255), 2)

            if count >= 4 and count <=9 and not alarm_on:
                alarm_on = True
                Thread(target=sound_manager.play_sound, args=("data/wake_up_alarm.mp3", True)).start()

             # Play msg.mp3 if count reaches 10
            if count == 10:
                sound_manager.play_sound("data/msg.mp3", False)  # Direct call, ensuring full playback

        else:
            count = 0
            cv2.putText(frame, "Eyes Open", (10, 48), cv2.FONT_ITALIC, 0.9, (0, 255, 0), 2)
            if alarm_on:
                alarm_on = False
                Thread(target=sound_manager.stop_sound).start()

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
