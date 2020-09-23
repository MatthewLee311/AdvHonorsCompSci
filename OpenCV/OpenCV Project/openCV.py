from __future__ import print_function
import argparse
import cv2
import numpy as np

def makingHistogram(image):
    
def imageTranslation(image):
    # Making the transformation for the image
    translation = np.float32([[1,0,300], [0,1,300]])
    # Translating the image
    translated = cv2.warpAffine(image, translation, (image.shape[1], image.shape[0]))
    # Displaying
    cv2.imshow("Translated to bottom right", translated)
    cv2.waitKey(0)

def imageCropping(image):
    # Cropping and Displaying the cropped image
    crop = image[0:500, 0:500]
    cv2.imshow("Cropped Image", crop)
    cv2.waitKey(0)

def displayImage(image):
    # Displaying info
    print("height: {} pixels".format(image.shape[0]))
    print("width: {} pixels".format(image.shape[1]))

    # Displaying Image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.imwrite("rainforest.jpg", image)

def main():
    # Making the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    displayImage(image)
    imageCropping(image)
    imageTranslation(image)
    makingHistogram(image)
main()
