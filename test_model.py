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
model = YOLO("best.pt")

# Determine the device to use
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize counters and other variables
FLOUR_COUNT = 0
FLAG = True
FLOUR_COUNTED = 0
FLAG2 = True
XY_COUNT = []
old_x1 = 0
video_lock = threading.Lock()


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
            XY_COUNT[i] = (X, Y)  # Обновляем значение на месте


def generate_frames():
    global FLOUR_COUNT, FLAG, XY_COUNT, old_x1

    # Determine the video source
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

    # Initialize VideoWriter only if processing an uploaded video
    if isinstance(source, str):
        output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))
    else:
        output = None

    while True:
        # Считываем кадр с веб-камеры
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"FLOUR_COUNT - {FLOUR_COUNT}", (880, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Вставка номера
        put_count2(XY_COUNT, FLOUR_COUNT, frame)

        # Обнаружение объектов в кадре с помощью YOLO
        results = model.predict(frame, conf=0.6, device=device, verbose=False)

        # Рисуем bounding boxes и метки классов на кадре
        for result in results:
            boxes = result.boxes  # Получаем координаты bounding box
            for box in boxes:

                print(box)
                confidence = box.conf[0]  # Получаем уверенность

                if confidence > 0.7:  # Увеличенный порог уверенности

                    # Преобразуем координаты в целые числа
                    x1 = int(box.xyxy[0][0])
                    y1 = int(box.xyxy[0][1])
                    x2 = int(box.xyxy[0][2])
                    y2 = int(box.xyxy[0][3])

                    # Рисуем bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    in_rectangle1 = is_inside((240, 535), (640, 950), (x1, y1), (x2, y2))
                    in_rectangle2 = is_inside((345, 405), (650, 730), (x1, y1), (x2, y2))

                    if in_rectangle1 and not in_rectangle2:
                        print("1 and not 2")
                        print((x1, y1), (x2, y2))
                        if FLAG:
                            FLOUR_COUNT += 1
                            FLAG = False

                            # XY для номера муки
                            XY_COUNT.append((455, 730))

                    elif in_rectangle2 and not in_rectangle1:
                        print("2 and not 1")

                        FLAG = True

        if output:
            output.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
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
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        cv2.destroyAllWindows()
