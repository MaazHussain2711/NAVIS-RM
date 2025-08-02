from flask import Flask, render_template_string, Response
import cv2
import time
import mediapipe as mp
from pyfirmata import ArduinoMega, SERVO, util

# ───────── Arduino Setup ─────────
PORT = '/dev/ttyUSB0'
SERVO_PIN = 4
board = ArduinoMega(PORT)
util.Iterator(board).start()
board.digital[SERVO_PIN].mode = SERVO
board.digital[SERVO_PIN].write(90)
time.sleep(2)

# ───────── MediaPipe Setup ─────────
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ───────── Camera ─────────
cap = cv2.VideoCapture(0)
current_angle = 90

def map_face_x_to_angle(x, frame_width):
    return max(30, min(150, int((x / frame_width) * 120 + 30)))

def gen_frames():
    global current_angle
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        frame_height, frame_width = frame.shape[:2]

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * frame_width)
                y = int(bboxC.ymin * frame_height)
                w = int(bboxC.width * frame_width)
                h = int(bboxC.height * frame_height)

                x_center = x + w // 2
                angle = map_face_x_to_angle(x_center, frame_width)

                if abs(angle - current_angle) > 2:
                    board.digital[SERVO_PIN].write(angle)
                    current_angle = angle

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f"Angle: {angle}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ───────── Flask App ─────────
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Tracking Servo</title>
</head>
<body>
    <h1>Face ➜ Servo Control</h1>
    <div style="text-align:center;">
        <img src="{{ url_for('video_feed') }}" width="720" height="540">
    </div>
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

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        cap.release()
        board.digital[SERVO_PIN].write(90)
        print("🛑 Cleaned up.")
