import time
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets as datasets
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

img = cv2.imread('digits.jpg')
printedImg = cv2.imread('printedDigits.jpg')

imgH, imgW, imgC = img.shape
printedImgH, printedImgW, printedImgC = printedImg.shape

framesize = 200
widthLoops = int(imgW/framesize * 2)
heightLoops = int(imgH/framesize * 2)

digitsModel = tf.keras.models.load_model('saved_models/digitsModelNonSoftmax')

for q in range(heightLoops):
    for i in range(widthLoops):
        # setting up frames
        topLeftX = int(framesize/2 * i)
        topLeftY = int(framesize/2 * q)
        bottomLeftX = int(framesize/2 * (i+1))
        bottomRightY = int(framesize/2 * (q+1))
        cv2.rectangle(img,(topLeftX, topLeftY), (bottomLeftX, bottomRightY), (255, 0, 0), 1)
        roi = img[topLeftY:bottomRightY, topLeftX:bottomLeftX]

        roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
        tmp = np.zeros((1, roi.shape[2], roi.shape[0], roi.shape[0]), dtype=float)
        for i, row in enumerate(roi):
            for j, col in enumerate(row):
                for k, entry in enumerate(col):
                    tmp[0][k][i][j] = entry/255.0
        roi = roi.reshape(-1, 28, 28, 1)
        roiTensor = torch.Tensor(tmp)
        results = digitsModel.predict(roi)
        print(results)


cv2.imshow("Press any key to exit", img)
cv2.waitKey()

framesizeReceipt = 50
widthLoops = int(printedImgW/framesizeReceipt * 2)
heightLoops = int(printedImgH/framesizeReceipt * 2)

for q in range(heightLoops):
    for i in range(widthLoops):
        # setting up frames
        topLeftX = int(framesizeReceipt/2 * i)
        topLeftY = int(framesizeReceipt/2 * q)
        bottomLeftX = int(framesizeReceipt/2 * (i+1))
        bottomRightY = int(framesizeReceipt/2 * (q+1))
        cv2.rectangle(printedImg,(topLeftX, topLeftY), (bottomLeftX, bottomRightY), (255, 0, 0), 1)
        roi = printedImg[topLeftY:bottomRightY, topLeftX:bottomLeftX]

        roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
        tmp = np.zeros((1, roi.shape[2], roi.shape[0], roi.shape[0]), dtype=float)
        for i, row in enumerate(roi):
            for j, col in enumerate(row):
                for k, entry in enumerate(col):
                    tmp[0][k][i][j] = entry/255.0
        roi = roi.reshape(-1, 28, 28, 1)
        roiTensor = torch.Tensor(tmp)
        results = digitsModel.predict(roi)
        print(results)


cv2.imshow("Press any key to exit", printedImg)
cv2.waitKey()
