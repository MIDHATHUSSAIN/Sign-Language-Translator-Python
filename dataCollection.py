import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imageSize = 300

folder = "Data/D"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imageWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imageSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imageSize))
            imageResizeShap = imgResize.shape
            wGap = math.ceil((imageSize - wCal) / 2)
            imageWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imageResizeShap = imgResize.shape
            hGap = math.ceil((imageSize - hCal) / 2)
            imageWhite[hGap:hCal + hGap, :] = imgResize
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Images_{time.time()}.jpg', imageWhite)
        print(counter)

    # Add a condition to break the loop
    if key == ord("Q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()



# def check_camera(index):
#     cap = cv2.VideoCapture(index)
#     if cap.isOpened():
#         print(f"Camera found at index {index}")
#         cap.release()
#     else:
#         print(f"No camera found at index {index}")
#
# # Check indices from 0 to 5
# for i in range(16):
#     check_camera(i)

