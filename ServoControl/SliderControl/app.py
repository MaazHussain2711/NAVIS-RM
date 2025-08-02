from flask import Flask, render_template, request, jsonify
from pyfirmata import ArduinoMega, SERVO, util

app = Flask(__name__)

# ───── Arduino Setup ─────
PORT = '/dev/ttyUSB0'  # Update if needed
board = ArduinoMega(PORT)
util.Iterator(board).start()

# Servo configuration (pin and range)
servos = {
    "bicep_left": {"pin": 2, "min": 30, "max": 150},
    "bicep_right": {"pin": 3, "min": 30, "max": 150},
    "eyes": {"pin": 5, "min": 0, "max": 180}, 
    "fingers_left": {"pin": 6, "min": 0, "max": 180},
    "fingers_right": {"pin": 8, "min": 0, "max": 180},
    "head": {"pin": 4, "min": 30, "max": 150},
    "wrist_left": {"pin": 9, "min": 0, "max": 90},
    "wrist_right": {"pin": 10, "min": 0, "max": 90}
}

# Set mode and initial position
for servo in servos.values():
    board.digital[servo["pin"]].mode = SERVO
    board.digital[servo["pin"]].write((servo["min"] + servo["max"]) // 2)

@app.route('/')
def index():
    return render_template('index.html', servos=servos)

@app.route('/move_servo', methods=['POST'])
def move_servo():
    data = request.json
    name = data.get('name')
    angle = int(data.get('angle'))
    if name in servos:
        pin = servos[name]['pin']
        board.digital[pin].write(angle)
        return jsonify({"status": "success", "servo": name, "angle": angle})
    return jsonify({"status": "error", "message": "Invalid servo"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
