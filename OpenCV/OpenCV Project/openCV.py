from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def treeDetection(image):
    # Detecting the trees using the canny edge detection
    edgeDetector = cv2.Canny(image, 50, 500)
    cv2.imshow("Tree Outlines", edgeDetector)
    cv2.waitKey(0)

def imageBlurring(image):
    # Doing average blurring with a 6x6 sized kernel
    blurring = np.hstack([
        cv2.blur(image, (6, 6))])
    cv2.imshow("Blurred Image", blurring)
    cv2.waitKey(0)
    return image

def makingHistogram(image):
    # Splitting Image into color channels
    channels = cv2.split(image)
    colors = ("b", "g", "r")
    # Making Histogram Graph
    plt.figure()
    plt.title("Rainforest Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    # Looping over the channels, making a histogram for each color channel
    for (chan, color) in zip(channels, colors):
        colorHistogram = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(colorHistogram, color = color)
        plt.xlim([0, 256])
    print("Close the histogram window by clicking the red 'X' in order to continue to the next step.")
    plt.show()


def imageTranslation(image):
    # Making the transformation for the image
    translation = np.float32([[1,0,300], [0,1,300]])
    # Translating the image
    imageTranslated = cv2.warpAffine(image, translation, (image.shape[1], image.shape[0]))

    # Rotating Image after the translation
    (h,w) = imageTranslated.shape[:2]
    center = (w // 2, h // 2)
    rotation = cv2.getRotationMatrix2D(center, 70, 1.0)
    rotatedImage = cv2.warpAffine(imageTranslated, rotation, (w,h))

    # Displaying
    cv2.imshow("Translated to bottom right & rotated 70 degrees", rotatedImage)
    cv2.waitKey(0)

def imageCropping(image):
    # Cropping and Displaying the cropped image
    crop = image[100:500, 200:700]
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

    # Calling functions to do things to the image
    displayImage(image)
    imageCropping(image)
    imageTranslation(image)
    makingHistogram(image)
    blurredImage = imageBlurring(image)
    treeDetection(blurredImage)
main()
