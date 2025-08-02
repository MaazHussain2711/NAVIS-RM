from flask import Flask, render_template_string, Response
import cv2

app = Flask(__name__)

# Load class names
classFile = r'/home/robo/NAVIS/ComputerVision/ObjectDetection/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the model
configPath = r'/home/robo/NAVIS/ComputerVision/ObjectDetection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'/home/robo/NAVIS/ComputerVision/ObjectDetection/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Generator for video stream
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read frame")
            break

        classIds, confs, bbox = net.detect(frame, confThreshold=0.55)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = classNames[classId - 1].upper()
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, className, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ Frame encoding failed")
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    print("✅ Camera released")

# HTML Page Template
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Object Detection Stream</title>
</head>
<body>
    <h2 style="text-align:center;">Real-Time Object Detection</h2>
    <div style="text-align:center;">
        <img src="{{ url_for('video') }}" width="640" height="480">
    </div>
</body>
</html>
"""

# Routes
@app.route('/')
def index():
    return render_template_string(html_page)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
