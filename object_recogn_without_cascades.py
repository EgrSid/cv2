# РАСПОЗНАВАНИЕ НОМЕРОВ БЕЗ КАСКАДОВ

import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt
import easyocr

img = cv.imread('img/car.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_filter = cv.bilateralFilter(gray, 11, 15, 15)
# 11 - диаметр фильтра (11px)
# 15 - количество цветов, которые будут смешиваться
# 15 (второе 15) - количество пикселей
edges = cv.Canny(img_filter, 30, 200)  # картинка с выделенными краями (грубо говоря перевод в векторный вид)

cont = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # ищем контуры по методу ветвления
cont = imutils.grab_contours(cont)  # доработка контуров
cont = sorted(cont, key=cv.contourArea, reverse=True)[:10]  # сортируем список по площади контура

pos = None
for c in cont:
    approx = cv.approxPolyDP(c, 10, True)  # чем больше epsision, тем ближе к квадрату
    if len(approx) == 4:
        pos = approx
        break

print(pos)

mask = np.zeros(gray.shape, dtype=np.uint8)
new_img = cv.drawContours(mask, [pos], 0, (255, 255, 255), -1)
bitwise_img = cv.bitwise_and(img, img, mask=mask)

x, y = np.where(mask == 255)  # все координаты, где есть белый цвет (где номер)
x1, y1 = np.min(x), np.min(y)  # верхняя левая точка номера
x2, y2 = np.max(x), np.max(y)  # нижняя правая точка номера

crop = gray[x1:x2, y1:y2]  # вырезаем номер машины с серого фото

text = easyocr.Reader(['en'])
text = text.readtext(crop)  # считываем сам номер
print(text)  # среди элементов списка есть считанный номер

plt.imshow(cv.cvtColor(crop, cv.COLOR_BGR2RGB))
plt.show()


