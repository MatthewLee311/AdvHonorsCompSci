# Python 3.8.6
# TENSOR FLOW NEEDS TO BE 3.6-3.8, 3.9 DOESNT WORK
# VI use opencv tensorflow, matplotlib, sklearn
# torch 1.7.0
# torchvision 0.8.1
# matplotlib 3.3.3
# numpy 1.19.4
# opencv-python 4.4.0
import time
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets as datasets
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

fout = open('test.txt', 'w')
now = time.strftime("%H:%M:%S", time.localtime())
print("[TIMER] Process Time:", now)
print("[TIMER] Process Time:", now, file = fout, flush = True)

# File location to save to or load from
MODEL_SAVE_PATH = './cifar_net.pth'
# Set to zero to use above saved model
TRAIN_EPOCHS = 20
# If you want to save the model at every epoch in a subfolder set to 'True'
SAVE_EPOCHS = False
# If you just want to save the final output in current folder, set to 'True'
SAVE_LAST = False
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4

devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    print('[INFO] GPU is detected.')
    print('[INFO] GPU is detected.', file = fout, flush = True)
else:
    print('[INFO] GPU not detected.')
    print('[INFO] GPU not detected.', file = fout, flush = True)

print("[INFO] Done importing packages.")
print("[INFO] Done importing packages.", file=fout)

class Net():
    def __init__(self, input_shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(12, 5, input_shape = input_shape, activation = 'relu'))
        self.model.add(layers.MaxPooling2D(pool_size = 2))
        self.model.add(layers.Conv2D(24, 5, activation = 'relu'))
        self.model.add(layers.MaxPooling2D(pool_size = 2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(160, activation = 'relu'))
        self.model.add(layers.Dense(50, activation = 'relu'))
        self.model.add(layers.Dense(10))
        #self.model.add(layers.Softmax())
        self.optimizer = optimizers.SGD(lr=0.001, momentum=0.9)
        self.loss = losses.MeanSquaredError()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)
        print(summaryStr, file=fout)

print("[INFO] Loading Traning and Test Datasets.")
print("[INFO] Loading Traning and Test Datasets.", file=fout)

# Get the CIFAR-10 Dataset.
#((trainX, trainY), (testX, testY)) = datasets.cifar10.load_data()
((trainX, trainY), (testX, testY)) = datasets.mnist.load_data()
#print(trainX.shape)
trainX = trainX.reshape(-1, 28, 28, 1)
#print(trainX.shape)
testX = testX.reshape(-1, 28, 28, 1)
# Convert from integers 0-255 to decimals 0-1.
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# Convert labels from integers to vectors.
lb = preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

net = Net((28, 28, 1))
print(net)

results = net.model.fit(trainX, trainY, validation_data=(testX, testY), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1)

plt.figure()
plt.plot(np.arange(0, 20), results.history['loss'])
plt.plot(np.arange(0, 20), results.history['val_loss'])
plt.plot(np.arange(0, 20), results.history['accuracy'])
plt.plot(np.arange(0, 20), results.history['val_accuracy'])
plt.show()

#Testing program on actual handwritten digits
net.model.save('saved_models/digitsModelNonSoftmax')
