import time
from pyfirmata import Arduino, util

# Replace with the correct port for your UNO (e.g. /dev/ttyUSB0 or /dev/ttyUSB1)
PORT = '/dev/ttyUSB1'
board = Arduino(PORT)
util.Iterator(board).start()
time.sleep(2)  # Allow board to initialize

# Define motor pins
# Front Right
FR_IN1 = board.digital[4]
FR_IN2 = board.digital[5]
# Front Left
FL_IN3 = board.digital[7]
FL_IN4 = board.digital[6]
# Rear Right
RR_IN3 = board.digital[12]
RR_IN4 = board.digital[13]
# Rear Left
RL_IN1 = board.digital[10]
RL_IN2 = board.digital[11]

motor_pins = [FR_IN1, FR_IN2, FL_IN3, FL_IN4, RR_IN3, RR_IN4, RL_IN1, RL_IN2]
for pin in motor_pins:
    pin.mode = 1  # Set all motor pins as OUTPUT

def stop_all():
    for pin in motor_pins:
        pin.write(0)

def forward():
    print("Moving FORWARD")
    FR_IN1.write(1); FR_IN2.write(0)
    # FL_IN3.write(1); FL_IN4.write(0)
    # RR_IN3.write(1); RR_IN4.write(0)
    # RL_IN1.write(1); RL_IN2.write(0)
    time.sleep(10)
    stop_all()

# Run forward movement
forward()
print("Forward movement completed.")
