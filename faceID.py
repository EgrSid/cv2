# ОБРАБОТКА ВИДЕО С КАМЕРЫ ПК + ОБРАБОТКА ФОНА ИЗОБРАЖЕНИЯ

import cv2 as cv
import tensorflow as tf
import numpy as np

face_cascade = cv.CascadeClassifier(
    'opencv-4.x/data/haarcascades/haarcascade_frontalcatface.xml')

capture = cv.VideoCapture(fr"video\my_test_vid.mp4")
capture.set(cv.CAP_PROP_FPS, 20)

model = tf.keras.models.load_model(fr"C:\projects\Python\ML\Models\FaceID\face_recogn_model_prep_681.h5")

i = 0
class_names = ['me', 'others']
while True:
    i += 1
    success, img = capture.read()
    try:
        faces = face_cascade.detectMultiScale(img,
                                              scaleFactor=1.1,
                                              minNeighbors=10,
                                              minSize=(50,
                                                       50))

        for x, y, w, h in faces:
            face = img[y:y + h, x:x + w]
            gray_face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            face_filter = cv.bilateralFilter(gray_face, 11, 15, 15)
            face = cv.Canny(face_filter, 20, 40)
            face = cv.resize(face, (681, 681))
            face = cv.cvtColor(face, cv.COLOR_GRAY2BGR)  # возвращаем глубину цвета 3

            img_array = tf.keras.utils.img_to_array(face)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            if class_names[np.argmax(score)] == 'me':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv.rectangle(img,
                         (x, y),
                         (x + w, y + h),
                         color,
                         4)
            cv.putText(img, f'{class_names[np.argmax(score)]}({round(np.max(score) * 100, 2)}%)',
                       (x, y - 20), 4, 1.0, color)
        img = cv.resize(img, (720, 1080))
        cv.imshow('Main Camera', img)
    except:
        pass

    key = cv.waitKey(30) & 0xFF
    if key == 27: break

capture.release()
cv.destroyAllWindows()
