import time
from pyfirmata import ArduinoMega, SERVO, util

# === Setup ===
PORT = '/dev/ttyUSB0'     # Change to correct port if needed
SERVO_PIN = 4          # Connected signal pin
DEBUG = True              # Enable verbose debug output

def debug(msg):
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

try:
    debug("🔌 Connecting to Arduino Mega...")
    board = ArduinoMega(PORT)
    util.Iterator(board).start()
    time.sleep(1)

    debug(f"⚙️ Setting pin {SERVO_PIN} to SERVO mode...")
    board.digital[SERVO_PIN].mode = SERVO
    time.sleep(1)

    debug("✅ Starting sweep test")

    while True:
        # Sweep 0 → 180
        for angle in range(30, 151, 10):
            debug(f"↗️ Moving to {angle}°")
            board.digital[SERVO_PIN].write(angle)
            time.sleep(0.5)

        # Sweep 180 → 0
        for angle in range(150, 29, -10):
            debug(f"↘️ Moving to {angle}°")
            board.digital[SERVO_PIN].write(angle)
            time.sleep(0.5)

except Exception as e:
    debug(f"❌ ERROR: {e}")

finally:
    debug("🛑 Setting servo to neutral (90°) and exiting.")
    board.digital[SERVO_PIN].write(90)
    time.sleep(1)
