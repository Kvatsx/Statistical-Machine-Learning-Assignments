# Kaustav Vats (2016048)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import ReadData

# Neural Netwrok ---------------------------------
class NeuralNetwork:

    def __init__(self, ncx, nco, nch1=256, nch2=128, nch3=64, nch4=0, actFunc="relu", learning_rate=0.1, epochs=100, verbose=True):
        self.ActFunc = actFunc
        self.LearningRate = learning_rate
        self.Epochs = epochs
        self.W1 = np.random.rand(ncx, nch1)
        self.B1 = np.random.rand(1, nch1)
        self.W2 = np.random.rand(nch1, nch2)
        self.B1 = np.random.rand(1, nch3)
        self.W3 = np.random.rand(nch2, nch3)
        self.B1 = np.random.rand(1, nch3)
        self.Output = np.random.rand(nch3, nco)
        if nch4 != 0:
            self.W4 = np.random.rand(nch3, nch4)
            self.B1 = np.random.rand(1, nch4)
            self.Output = np.random.rand(nch4, nco)
        if verbose:
            print("Activation Function:\t{}".format(self.ActFunc))
            print("Learning Rate:\t{}".format(self.LearningRate))
            print("Epochs:\t{}".format(self.Epochs))
            print("Input Layer:\t{}".format(ncx))
            print("Layer 1:\t{}".format(self.W1.shape))
            print("Layer 2:\t{}".format(self.W2.shape))
            print("Layer 3:\t{}".format(self.W3.shape))
            if nch4 != 0:
                print("Layer 4:{}".format(self.W4.shape))
            print("Output Layer:{}".format(self.Output.shape))

    def Sigmoid(self, zx):
        ret = 1/(1 + np.exp(-zx))
        return ret
    
    def Relu(self, zx):
        if zx > 0:
            return zx
        return 0

    def FeedForward(self, x_data, actFunc="relu"):
        Z1 = np.dot(x_data, self.W1) + self.B1
        self.A1 = self.Sigmoid(Z1)
        Z2 = np.dot(self.A1, self.W2) + self.B1
        self.A2 = self.Sigmoid(Z2)
        Z3 = np.dot(self.A2, self.W3) + self.B1
        self.A3 = self.Sigmoid(Z3)

    '''Updating weights of all neurons after training'''
    def BackPropogation(self):




if __name__ == "__main__":
    '''Reading Mnist Dataset'''
    X_Train, Y_Train, X_Test, Y_Test = ReadData(verbose=True)
    print("------------------------------------------------")

    NN = NeuralNetwork(X_Train.shape[0], np.unique(Y_Train).shape[0], nch4=0)
