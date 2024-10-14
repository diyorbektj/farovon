import cv2

# Функция проверки нахождения одного прямоугольника внутри другого
def is_inside(rect1_start, rect1_end, rect2_start, rect2_end):
    return (rect2_start[0] >= rect1_start[0] and rect2_start[1] >= rect1_start[1] and
            rect2_end[0] <= rect1_end[0] and rect2_end[1] <= rect1_end[1])


def put_count(XY_COUNT, frame):
    for i in range(len(XY_COUNT)):
        X, Y, KEY = XY_COUNT[i]
        if Y > 20:
            Y += 1
            X -= 45
            cv2.putText(frame, f"{KEY}", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            XY_COUNT[i] = (X, Y, KEY)  # Обновляем значение на месте

