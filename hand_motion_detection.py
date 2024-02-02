import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
prev_frame = None
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            if (
                prev_frame and 
                (abs(prev_frame.x-results.multi_hand_landmarks[0].landmark[0].x)**2+abs(prev_frame.y-results.multi_hand_landmarks[0].landmark[0].y)**2)**(.5) >= 0.05
            ):
                cv2.putText(
                image,
                "Hello",
                (50,100),
                font, 
                1,
                (0, 255, 255),
                2,
                cv2.LINE_4
            )
            else:
                prev_frame = results.multi_hand_landmarks[0].landmark[0]
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            cv2.putText(
                image,
                "Hand Detected",
                (50,50),
                font, 
                1,
                (0, 255, 255),
                2,
                cv2.LINE_4
            )
        else:
            prev_frame = None
        image = cv2.flip(image, 1)
        cv2.imshow('Hand Detection', image)
        if (cv2.waitKey(5) & 0xFF == ord('q')):
            break
cap.release()
cv2.destroyAllWindows()