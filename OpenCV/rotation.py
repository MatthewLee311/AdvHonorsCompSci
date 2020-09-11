import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

(h,w) = image.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, 25, 2.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("Rotated by 45 degrees", rotated)

M = cv2.getRotationMartix2D(center, -270, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("Rotated by -270 degrees", rotated)
