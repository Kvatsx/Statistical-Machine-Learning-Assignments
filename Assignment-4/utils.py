# Kaustav Vats

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Data.MNIST.utils import mnist_reader

'''Read MNIST Data'''
def ReadData(path="./Data/MNIST/", verbose=True):
    x_train, y_train = mnist_reader.load_mnist(path, kind='train')
    x_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
    if verbose:
        print("X_Train.shape:{},\tY_Train.shape:{}".format(x_train.shape, y_train.shape))
        print("X_Test.shape:{},\tY_Test.shape:{}".format(x_test.shape, y_test.shape))
        print("No of classes:{}".format(np.unique(y_test).shape))
    return x_train, y_train, x_test, y_test


