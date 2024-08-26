# ОБРАБОТКА ВИДЕО С КАМЕРЫ ПК + ОБРАБОТКА ФОНА ИЗОБРАЖЕНИЯ

import cv2 as cv
import numpy as np

from cvzone.SelfiSegmentationModule import SelfiSegmentation  # обученная модель от tf

img_bg = cv.imread(fr'img\haha.jpg')

face_cascade = cv.CascadeClassifier(
    'opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')  # подключили модель для распознавания лиц
segmentor = SelfiSegmentation()

capture = cv.VideoCapture(0)  # 0 - дефолтная камера компа


def blur(img, k=3):
    """Гауссово размытие"""
    h, w = img.shape[:2]  # сначала высота, потом ширина (там есть еще элементы, поэтому берем только первые 2)

    dw, dh = int(w / k), int(h / 3)
    if dw % 2 == 0: dw -= 1
    if dh % 2 == 0: dh -= 1

    sigma = 0  # стандартное отклонение
    return cv.GaussianBlur(img, (dw, dh), sigma)


while True:
    success, img = capture.read()  # захват кадра

    # ИЗМЕНЕНИЕ ФОНА ВИДЕО
    # шейпы должны быть одинаковые у фона и оставшейся части картинки, на которую накладываем
    img_bg = cv.resize(img_bg, (img.shape[1], img.shape[0]))
    img = segmentor.removeBG(img, img_bg, cutThreshold=0.8)  # cutThreshold - точность обрезки

    # возвращает все, что находит (все лица)
    faces = face_cascade.detectMultiScale(img,
                                          scaleFactor=1.4,  # увеличиваем область выделения лица
                                          minNeighbors=5,
                                          minSize=(20,
                                                   20))  # если лицо на картинке меньше заданного размера, то его не будем рассматривать

    # БЛЮР ЛИЦА
    for x, y, w, h in faces:
        cv.rectangle(img,
                     (x, y),  # верхний левый угол
                     (x + w, y + h),  # нижний правый угол
                     (255, 0, 0),  # RGB, но только BGR
                     2)  # толщина границ

        img[y:y + h, x:x + w] = blur(img[y:y + h, x:x + w])  # блюрим лицо внутри квадратика

    cv.imshow('Main Camera', img)  # подпись к окну (только английские буквы), само изображение
    # к этому окну потом можно обращаться через название

    key = cv.waitKey(30) & 0xFF  # выход из цикла, чтобы можно было выйти из просмотра изображения
    if key == 27: break  # 27 - esc

capture.release()  # убираем ресурсы из мониторинга для слежения за камерой
cv.destroyAllWindows()  # закрываем все созданные окна
