import cv2 as cv

vid = cv.VideoCapture('video/registrator.mp4')
vid.set(cv.CAP_PROP_FPS, 30)

cascade = cv.CascadeClassifier("opencv-4.x\data\haarcascades\haarcascade_russian_plate_number.xml")

while True:
    success, img = vid.read()

    sh = img.shape
    img = cv.resize(img, (int(sh[1] // 1.5), int(sh[0] // 1.5)))

    nums = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(15, 15))

    for x, y, w, h in nums:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow('main', img)

    if cv.waitKey(20) & 0xFF == 27: break

vid.release()
cv.destroyAllWindows()
