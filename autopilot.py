import cv2 as cv
import numpy as np


def canny(img):
    """перевод в серый формат + размытие + выделение контуров"""
    img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    blur = cv.GaussianBlur(img, (9, 9), 0)
    return cv.Canny(blur, 50, 100)  # предпочтительно отношение 1:2 или 1:3


def make_coordinates(image, line_parameters):
    k, b = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 3/5)
    x1 = int((y1 - b) / k)
    x2 = int((y2 - b) / k)
    return np.array([x1, y1, x2, y2])


def mask(image):
    """создаем маску трапеции и накладываем ее на изоблражение"""
    polygons = np.array(POLYGON)  # трапеция
    mask = np.zeros_like(image)
    cv.fillPoly(mask, np.array([polygons], dtype=np.int64), 1024)  # наложили трапецию на черный фон
    masked_img = cv.bitwise_and(image, mask)  # совместили трапецию и изображение
    return masked_img


def average_b_k(image, lines):
    left_fit, right_fit = [], []
    while lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            b = parameters[0]
            k = parameters[1]
            if b < 0:
                left_fit.append((b, k))
            else:
                right_fit.append((b, k))
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(line_image, (x1, y1), (x2, y2),
                    (255, 255, 255), 15)

    return line_image

def region_of_interest(image):
    polygons = np.array(POLYGON)
    mask = np.zeros_like(image)
    cv.fillPoly(mask, np.array([polygons], dtype=np.int64), 1024)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image



video = cv.VideoCapture('video/test_autopilot.mp4')

if not video.isOpened():
    print('ERROR! Video has opened!!!')

while video.isOpened():
    _, frame = video.read()
    copy_img = np.copy(frame)

    try:
        width = frame.shape[1]
        height = frame.shape[0]
        POLYGON = [(0 + width // 4, height // 2), (width - width // 4, height // 2),
                   (width - width // 10, height // 5 * 4), (0 + width // 10, height // 5 * 4)]
        frame = mask(frame)
        frame = canny(frame)
        frame = region_of_interest(frame)
        lines = cv.HoughLinesP(frame, 2, np.pi / 180, 100,
                               np.array([()]),
                               minLineLength=20, maxLineGap=5)
        average_lines = average_b_k(frame, lines)

        line_image = display_lines(copy_img, average_lines)

        combo = cv.addWeighted(copy_img, 0.8, line_image, 0.5, 1)
        cv.imshow('Video', combo)
    except:
        pass

    if cv.waitKey(1) & 0xFF == 27:
        video.release()
        cv.destroyAllWindows()

video.release()
cv.destroyAllWindows()
