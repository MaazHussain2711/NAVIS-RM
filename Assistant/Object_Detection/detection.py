import cv2
import threading
import time
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

last_spoken = {}
cooldown = 2  # seconds

# Load class names
classFile = "/home/robo/raspi_arduino_testing/Voice_Assistant/Test/Object_Detection/coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model files
configPath = "/home/robo/raspi_arduino_testing/Voice_Assistant/Test/Object_Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/robo/raspi_arduino_testing/Voice_Assistant/Test/Object_Detection/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Speak function using espeak
def speak(text):
    print(f"🔊 Speaking: {text}")
    os.system(f'espeak "{text}" --stdout | aplay')

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        classIds, confs, bbox = net.detect(img, confThreshold=0.55)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = classNames[classId - 1].upper()
                print(f"[OBJECT] Detected {className} - {round(confidence * 100, 2)}%")

                now = time.time()
                if className not in last_spoken or now - last_spoken[className] > cooldown:
                    last_spoken[className] = now
                    speak(f"Detected {className}")

        key = cv2.waitKey(1)
        if key == ord('q'):
            speak("Exiting object detection.")
            break
except KeyboardInterrupt:
    print("Object detection interrupted.")
finally:
    cap.release()
    cv2.destroyAllWindows()
