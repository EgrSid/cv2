import cv2 as cv

vid = cv.VideoCapture(fr"C:\ML\DataBases\faces\new_face_data\egor_new.mp4")

i = 0
while True:
    i += 1
    success, img = vid.read()
    img = cv.resize(img, (500, 500))
    cv.imwrite(f'C:/ML/DataBases/faces/face_data_500/me/im{i}.jpg', img)

    cv.imshow('Main Camera', img)

    key = cv.waitKey(30) & 0xFF
    if key == 27: break

vid.release()
cv.destroyAllWindows()
