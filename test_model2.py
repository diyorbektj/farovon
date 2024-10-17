import os
import cv2
import torch
from ultralytics import YOLO
from flask import Flask, Response, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from functions import is_inside, put_count  # Ensure this function is correctly defined in your functions.py
import threading

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for flashing messages

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained YOLO model
model = YOLO("best4.pt")

# Determine the device to use
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
if device == "cuda":
    model = model.half()
# Initialize counters and other variables
FLOUR_COUNT = 0
FLAG = True
FLOUR_COUNTED = 0
FLAG2 = True
XY_COUNT = []
old_x1 = 0
video_lock = threading.Lock()


class VideoStream:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            with self.lock:
                self.ret, self.frame = self.cap.read()

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def put_count2(XY_COUNT, FLOUR_COUNT, frame):
    for i in range(len(XY_COUNT)):
        X, Y = XY_COUNT[i]
        if Y > 20:
            Y -= 9
            X += 2
            cv2.putText(frame, f"{FLOUR_COUNT}", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            XY_COUNT[i] = (X, Y)  # Update the value in place


def generate_frames():
    global FLOUR_COUNT, FLAG, XY_COUNT, old_x1

    # Determine the video source
    with video_lock:
        if 'VIDEO_PATH' in app.config and app.config['VIDEO_PATH']:
            source = app.config['VIDEO_PATH']
            cap = cv2.VideoCapture(source)
        else:
            source = 0  # Webcam
            cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30  # Default to 30 if FPS not available
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    # Initialize VideoWriter only if processing an uploaded video
    if isinstance(source, str):
        output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))
    else:
        output = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection using YOLO
        results = model.predict(frame, conf=0.6, device=device, verbose=False)

        cv2.putText(frame, f"FLOUR_COUNT - {FLOUR_COUNT}", (880, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Insert count number
        put_count(XY_COUNT, frame)
        print(results)

        # Draw bounding boxes and class labels
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box
            cv2.rectangle(frame, (x1 - 100, y1), (x2 + 50, y2), (0, 255, 0), 3)

            if x1 > old_x1 and x1 > 1000:
                FLOUR_COUNT += 1
                XY_COUNT.append((1200, 350, FLOUR_COUNT))
            old_x1 = x1

        # Write frame to output video if applicable
        if output:
            output.write(frame)

        # Encode frame as JPEG

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # Reduce quality to 50

        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release resources
    cap.release()
    if output:
        output.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.form.get('video_url'):
        app.config['VIDEO_PATH'] = request.form.get('video_url')
        flash('Video successfully uploaded')
        return redirect(url_for('index'))
    if 'video_file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['video_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        # Update the VIDEO_PATH in app config
        with video_lock:
            app.config['VIDEO_PATH'] = video_path
        flash('Video successfully uploaded')
        return redirect(url_for('index'))
    else:
        flash('Allowed file types are mp4, avi, mov, mkv')
        return redirect(url_for('index'))


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=False)
    finally:
        cv2.destroyAllWindows()

