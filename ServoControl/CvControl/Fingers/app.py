import cv2
import threading
import time
import io
from flask import Flask, render_template_string, Response
from pyfirmata import ArduinoMega, util
import mediapipe as mp

# ───────── Arduino Setup ─────────
board = ArduinoMega('/dev/ttyUSB0')  # Adjust if needed
util.Iterator(board).start()
LEFT_PIN = 6
RIGHT_PIN = 8
board.digital[LEFT_PIN].mode = 1  # OUTPUT
board.digital[RIGHT_PIN].mode = 1

# ───────── MediaPipe Setup ─────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
FINGER_TIPS = [4, 8, 12, 16, 20]

# ───────── Global Variables ─────────
left_status = "Not Detected"
right_status = "Not Detected"
frame_lock = threading.Lock()
output_frame = None

# ───────── Flask App ─────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Hand Detection</title>
    <style>
        body { background-color: #111; color: white; font-family: Arial; text-align: center; }
        h1 { margin-top: 20px; }
        .status { font-size: 1.5em; margin: 10px; }
        img { border: 2px solid white; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>🖐️ Hand Detection Live Feed</h1>
    <img src="{{ url_for('video_feed') }}" width="640" height="480">
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, left=left_status, right=right_status)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ───────── Hand Status Helper ─────────
def hand_status(hand_landmarks, handedness):
    lm = hand_landmarks.landmark
    label = handedness.classification[0].label
    fingers = []

    if label == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)

    for tip_id in FINGER_TIPS[1:]:
        fingers.append(1 if lm[tip_id].y < lm[tip_id - 2].y else 0)

    is_open = sum(fingers) >= 4
    return label, "Open" if is_open else "Closed"

# ───────── Video Processing Thread ─────────
def video_loop():
    global output_frame, left_status, right_status

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left = right = "Not Detected"

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label, status = hand_status(hand_landmarks, handedness)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if label == "Left":
                    left = status
                elif label == "Right":
                    right = status

        left_status = left
        right_status = right

        # Send digital signal to Arduino based on hand state
        board.digital[LEFT_PIN].write(1 if left == "Open" else 0)
        board.digital[RIGHT_PIN].write(1 if right == "Open" else 0)

        # Overlay text
        cv2.putText(frame, f"Left Hand: {left}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Right Hand: {right}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Save current frame for streaming
        with frame_lock:
            output_frame = frame.copy()

    cap.release()

# ───────── Frame Generator for Flask ─────────
def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ───────── Start Server ─────────
if __name__ == "__main__":
    t = threading.Thread(target=video_loop)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=5000)
