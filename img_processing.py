# ОБРАБОТКА ИЗОБРАЖЕНИЯ

import cv2 as cv
import numpy as np
import imutils
from random import randrange

img = cv.imread('img\plate (2).jpg')

height, width, color_profile = img.shape

cv.imshow('main', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # перевод изображения в серый формат
cv.imshow('main', gray)
#  чтобы перевести в чб, надо сначала в серый, а уже потом в чб. КАЧЕСТВО БУДЕТ ЛУЧШЕ
black = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)[1]  # картинка, 0 - понижает цвет до нуля, 255 - максимальная яркость
cv.imshow('main', black)

# выделение контуров
edges = cv.Canny(black, 10, 250)  # контуры, состоящие из минимально 10 вершин, а максимально из 250 вершин
cv.imshow('main', edges)  # выделение контуров

# закрытие шумных контуров
# 1 СПОСОБ (немного лучше, просто меньше писать)
kernel = cv.getStructuringElement(cv.MORPH_RECT,  # кисть будет прямоугольная
                                  (3, 3))  # размер кисти
closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
cv.imshow('main', closed)

# 2 СПОСОБ
kernel = np.ones((3, 3), dtype=np.uint8)
closed = cv.dilate(edges, kernel, iterations=1)
closed = cv.erode(closed, kernel)  # работа с толщиной контуров
cv.imshow('main', closed)

# поиск контуров
# cntrs - замкнутая ломаная линия
# edges - не замкнутая ломаная линия
cntrs = cv.findContours(closed,
                        cv.RETR_EXTERNAL,  # экстраполяция
                        cv.CHAIN_APPROX_SIMPLE)  # аппроксимация
cntrs = imutils.grab_contours(cntrs)  # получили массив из контуров

k = 0
for c in cntrs:
    p = cv.arcLength(c, True)  # считаем длину контура (True - чтобы не бегал по кругу бесконечно)
    approx = cv.approxPolyDP(c, 0.02 * p, True)  # сглаживаем контуры на 2%

    if len(approx) == 4:  # если мы нашли контур номера (4 вершины)
        cv.drawContours(img, [approx], -1, (0, 255, 0), 3)
    k += 1
cv.imshow('main', img)
print(k)

cv.waitKey(0)  # жди нажатия любой кнопки бесконечно, иначе закрывай картинку (НЕ РАБОТАЕТ ДЛЯ ВИДЕО)


# ОБРАБОТКА КАРТИНКИ  2 ЧАСТЬ

photo = cv.imread('img\haha.jpg', cv.IMREAD_COLOR)

# cv.IMREAD_COLOR = -1
# cv.IMREAD_GRAYSCALE = 0
# cv.IMREAD_UNCHANGED = 1 (with alpha, отвечает за насыщенность картинки)

photo = cv.resize(photo, (photo.shape[0] // 4, photo.shape[1] // 4))
photo = cv.resize(photo, (0, 0), fx=0.9, fy=0.9)  # изменение размера картинки через коэффициенты масштабирования cv7
photo = cv.rotate(photo, cv.ROTATE_90_CLOCKWISE)  # разворот картинки ан 90 градусов

for i in range(100):  # меняем первые 100 пикселей по Y на случайные цвета
    for j in range(photo.shape[1]):
        photo[i][j] = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]

cv.imwrite('img\haha2.jpg', photo)  # записываем новое изображение

img = np.zeros(photo.shape[:2], dtype='uint8')

circle = cv.circle(img.copy(), (200, 300), 120, (255, 255, 255), -1)
rectangle = cv.rectangle(img.copy(), (200, 300), (250, 150), (255, 255, 255), -1)
line = cv.line(img.copy(), (0, 0), (int(img.shape[0]), int(img.shape[1])), (255, 255, 255), 4)

#  img = cv.bitwise_and(rectangle, circle)  # бинарное произведение окружности и прямоугольника (пересечение этих фигур)
#  img = cv.bitwise_and(rectangle, line)
#  img = cv.bitwise_or(rectangle, circle)  # объединяет прямоугольник и окружность
#  img = cv.bitwise_xor(rectangle, circle)  # логическое ИЛИ, но только при всех нулях возвращается единица
#  img = cv.bitwise_not(rectangle, circle)
img = cv.bitwise_and(photo, photo, mask=circle)

cv.imshow('haha', img)
cv.waitKey(0)
cv.destroyAllWindows()


# ОБРАБОТКА ЦВЕТА ИЗОБРАЖЕНИЯ

img = cv.imread('img\colored_lion.jpg')
b = g = r = 0
clicked = False


def color_fn(event, x, y, flags, param):
    global b, g, r, clicked
    if event == cv.EVENT_LBUTTONUP:
        b, g, r = img[y, x]  # получаем цвета в месте, куда наведена мышка
        clicked = True


cv.namedWindow('main')
cv.setMouseCallback('main', color_fn)
while True:
    cv.imshow('main', img)

    if clicked:
        print((b, g, r))
        cv.rectangle(img, (50, 50), (300, 100), (int(b), int(g), int(r)), -1)  # -1, чтобы залить квадрат
        cv.putText(img, f'rgb: {(r, g, b)}', (50, 100), 2, 1.0, (255, 255, 255))
        clicked = False

    if cv.waitKey(30) & 0xFF == 27: break

cv.destroyAllWindows()
