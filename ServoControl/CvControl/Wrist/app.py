from flask import Flask, render_template_string, Response
import cv2
import mediapipe as mp
import time
from pyfirmata import ArduinoMega, SERVO, util

# ─── Arduino Mega Setup ───
PORT = '/dev/ttyUSB0'  # Raspberry Pi USB port
LEFT_SERVO_PIN = 9
RIGHT_SERVO_PIN = 10

board = ArduinoMega(PORT)
util.Iterator(board).start()
board.digital[LEFT_SERVO_PIN].mode = SERVO
board.digital[RIGHT_SERVO_PIN].mode = SERVO
time.sleep(2)

left_angle = right_angle = 90
board.digital[LEFT_SERVO_PIN].write(left_angle)
board.digital[RIGHT_SERVO_PIN].write(right_angle)

# ─── Mediapipe Setup ───
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ─── Flask App ───
app = Flask(__name__)
cap = cv2.VideoCapture(0)

def gen_frames():
    global left_angle, right_angle
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_type = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
                lm = hand_landmarks.landmark
                index_x = lm[5].x * w  # Index MCP
                pinky_x = lm[17].x * w  # Pinky MCP
                diff = (index_x - pinky_x) / w
                diff = max(-0.10, min(0.10, diff))  # Clamp noise
                mapped_angle = int((diff + 0.10) / 0.20 * 180)

                if hand_type == "Left":
                    if abs(mapped_angle - left_angle) > 2:
                        board.digital[LEFT_SERVO_PIN].write(mapped_angle)
                        left_angle = mapped_angle
                    print(f"👈 Left Hand: {mapped_angle}°")
                    cv2.putText(frame, f"Left: {mapped_angle}°", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif hand_type == "Right":
                    if abs(mapped_angle - right_angle) > 2:
                        board.digital[RIGHT_SERVO_PIN].write(mapped_angle)
                        right_angle = mapped_angle
                    print(f"👉 Right Hand: {mapped_angle}°")
                    cv2.putText(frame, f"Right: {mapped_angle}°", (w - 180, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
        <!doctype html>
        <html lang="en">
        <head>
            <title>Dual Hand Servo Control</title>
            <style>
                body { text-align: center; background-color: #222; color: white; font-family: Arial; }
                h1 { margin-top: 20px; }
                img { border: 3px solid white; border-radius: 10px; margin-top: 10px; max-width: 90%; height: auto; }
            </style>
        </head>
        <body>
            <h1>👋 Dual Hand Servo Control</h1>
            <img src="{{ url_for('video_feed') }}">
        </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ─── Cleanup on exit ───
import atexit
@atexit.register
def cleanup():
    cap.release()
    board.digital[LEFT_SERVO_PIN].write(90)
    board.digital[RIGHT_SERVO_PIN].write(90)
    print("🧹 Cleaned up and reset servos.")

# ─── Run App ───
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
