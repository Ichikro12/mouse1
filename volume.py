import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

# Initialize volume range
min_volume = 0
max_volume = 100

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand_landmarks in hands:
            # Check if the hand is the left hand
            if hand_landmarks and output.multi_handedness[0].classification[0].label == 'Left':
                drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                # Detect middle and thumb fingers
                middle_finger = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

                # Convert normalized coordinates to pixel coordinates
                middle_x = int(middle_finger.x * frame_width)
                middle_y = int(middle_finger.y * frame_height)
                thumb_x = int(thumb.x * frame_width)
                thumb_y = int(thumb.y * frame_height)

                # Calculate distance between middle finger and thumb
                distance = math.sqrt((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2)

                # Map distance to volume range
                volume = int((distance / frame_width) * (max_volume - min_volume))

                # Ensure volume stays within range
                volume = max(min(volume, max_volume), min_volume)

                # Adjust system volume based on hand gesture
                if thumb_y < middle_y:
                    pyautogui.press('volumeup', presses=volume, interval=0.05)
                else:
                    pyautogui.press('volumedown', presses=volume, interval=0.05)

    cv2.imshow('VM', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
