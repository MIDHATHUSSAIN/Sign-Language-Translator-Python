import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imageSize = 300

folder = "Data/D"
counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    # imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(img)
            print(prediction, index)


        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imageResizeShap = imgResize.shape
            hGap = math.ceil((imageSize - hCal) / 2)
            imageWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(img)
        cv2.putText(img, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("Image", img)

    cv2.waitKey(1)




