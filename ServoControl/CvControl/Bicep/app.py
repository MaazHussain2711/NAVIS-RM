from flask import Flask, render_template_string, Response
import cv2
import math
import time
import mediapipe as mp
from pyfirmata import ArduinoMega, SERVO, util

# ───────── Arduino Setup ─────────
PORT = '/dev/ttyUSB0'  # ✅ Use correct Raspberry Pi port
LEFT_SERVO_PIN = 2
RIGHT_SERVO_PIN = 3

board = ArduinoMega(PORT)
util.Iterator(board).start()

board.digital[LEFT_SERVO_PIN].mode = SERVO
board.digital[RIGHT_SERVO_PIN].mode = SERVO
board.digital[LEFT_SERVO_PIN].write(90)
board.digital[RIGHT_SERVO_PIN].write(90)
time.sleep(2)

# ───────── Constants ─────────
ELBOW_ACTIVE_MIN = 120
ELBOW_ACTIVE_MAX = 180
SERVO_MIN = 30
SERVO_MAX = 150

# ───────── MediaPipe Pose Setup ─────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0)
drawer = mp.solutions.drawing_utils

# ───────── Camera ─────────
cap = cv2.VideoCapture(0)


def elbow_angle(a, b, c):
    """Returns ∠ABC given three points."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    return math.degrees(math.acos(max(-1, min(1, dot / mag)))) if mag else 0

def map_elbow_to_servo(angle_deg):
    """Maps elbow angle (120-180) to servo angle (180-0)"""
    angle_deg = max(ELBOW_ACTIVE_MIN, min(ELBOW_ACTIVE_MAX, angle_deg))
    norm = (angle_deg - ELBOW_ACTIVE_MIN) / (ELBOW_ACTIVE_MAX - ELBOW_ACTIVE_MIN)
    return int(SERVO_MIN + (1 - norm) * (SERVO_MAX - SERVO_MIN))

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Left arm
            L_SH = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            L_EL = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            L_WR = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            elb_left = elbow_angle(L_SH, L_EL, L_WR)
            servo_left = map_elbow_to_servo(elb_left)
            board.digital[LEFT_SERVO_PIN].write(servo_left)

            # Right arm
            R_SH = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            R_EL = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            R_WR = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            elb_right = elbow_angle(R_SH, R_EL, R_WR)
            servo_right = map_elbow_to_servo(elb_right)
            board.digital[RIGHT_SERVO_PIN].write(servo_right)

            # Draw overlay
            drawer.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Left Servo: {servo_left}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Right Servo: {servo_right}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ───────── Flask App ─────────
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dual Elbow Servo Control</title>
</head>
<body>
    <h1>Dual Elbow ➜ Servo Control (Live)</h1>
    <img src="{{ url_for('video_feed') }}" width="720" height="540">
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ───────── Run App ─────────
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        cap.release()
        board.digital[LEFT_SERVO_PIN].write(90)
        board.digital[RIGHT_SERVO_PIN].write(90)
        print("🛑 Cleaned up.")
