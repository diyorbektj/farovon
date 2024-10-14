import torch
from ultralytics import YOLO
from functions import *

# Загружаем обученную модель YOLO
model = YOLO("best4.pt")

# model.export(format='openvino')
url_video = input("Url video:")
# Инициализируем веб-камеру или видеофайл
cap = cv2.VideoCapture(url_video)  # Или укажите путь к видеофайлу
if not cap.isOpened():
    raise IOError("Не удалось открыть веб-камеру или видеофайл")

# Получаем параметры видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Инициализируем объект VideoWriter для записи видео
output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

FLOUR_COUNT = 0
FLAG = True

FLOUR_COUNTED = 0
FLAG2 = True

XY_COUNT = []
old_x1 = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обнаружение объектов в кадре с помощью YOLO
    results = model.predict(frame, conf=0.6, device=device, verbose=False)

    cv2.putText(frame, f"FLOUR_COUNT - {FLOUR_COUNT}", (880, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Вставка номера
    put_count(XY_COUNT, frame)
    print(results)
    # Рисуем bounding boxes и метки классов на кадре
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    classes = results[0].boxes.cls.cpu().numpy()

    for box, score, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = map(int, box)

        # Рисуем bounding box
        cv2.rectangle(frame, (x1-100, y1), (x2+50, y2), (0, 255, 0), 3)

        if x1 > old_x1 and x1 > 1000:
            FLOUR_COUNT += 1
            FLOUR_COUNTED += 1
            FLAG = False
            XY_COUNT.append((1200, 350, FLOUR_COUNT))
        old_x1 = x1




    # Записываем кадр в видеофайл
    output.write(frame)
    cv2.imshow("Image", frame)


    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
output.release()  # Важно освободить VideoWriter
cv2.destroyAllWindows()
