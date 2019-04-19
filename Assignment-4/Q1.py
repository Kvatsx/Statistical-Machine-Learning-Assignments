# Kaustav Vats (2016048)
# Ref:- [1] https://zhenye-na.github.io/2018/09/09/build-neural-network-with-mnist-from-scratch.html
# Ref:- [2] https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/
# Ref:- [3] https://enlight.nyc/projects/neural-network/
# Ref:- [4] http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# Ref:- [5] https://github.com/gary30404/neural-network-from-scratch-python/blob/master/network.py
# Ref:- [6] https://github.com/vzhou842/neural-network-from-scratch/blob/master/network.py
# Ref:- [7] https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
# Ref:- [8] https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import ReadData

# Neural Netwrok ---------------------------------
class NeuralNetwork:

    def __init__(self, input_size, nodes_hidden, verbose=True):
        self.InputSize = input_size
        self.LayersCount = len(nodes_hidden)
        self.W = []
        self.B = []
        self.Initialize(nodes_hidden)

        if verbose:
            print("------ Neural Network Structure -------")
            for i in range(self.LayersCount):
                print("[W] Layer {} shape: {}".format(i, self.W[i].shape))         
            for i in range(self.LayersCount):
                print("[B] Layer {} shape: {}".format(i, self.B[i].shape))

    def Initialize(self, node_hidden):
        prev = self.InputSize
        for i in range(self.LayersCount):
            now = node_hidden[i]
            self.W.append(np.random.randn(prev, now)/np.sqrt(now))
            self.B.append(np.random.randn(1, now)/np.sqrt(now))
            prev = now

    def Sigmoid(self, z):
        ret = 1/(1 + np.exp(-z))
        return ret
    
    def Relu(self, z):
        if z > 0:
            return z
        return 0

    def Sigmoid_Derivative(self, z):
        ret = z * (1 - z)
        return ret
    
    def Softmax(self, z):
        Exp = np.exp(z)
        Total = np.sum(Exp, axis=1)
        return Exp / Total

    def FeedForward(self, x_data, actFunc="relu"):
        self.A = []
        Activation = x_data
        for i in range(self.LayersCount):
            z = np.dot(Activation, self.W[i]) + self.B[i]
            A = self.Sigmoid(z)
            self.A.append(A)

    '''Updating weights of all neurons after training'''
    def BackPropogation(self, y_data, output):
        self.E = y_data - output
        # self.

    def saveModel(self):
        for i in range(self.LayersCount):
            np.save("./Data/W{}.npy".format(i), self.W[i])
            np.save("./Data/B{}.npy".format(i), self.B[i])        



if __name__ == "__main__":
    '''Reading Mnist Dataset'''
    X_Train, Y_Train, X_Test, Y_Test = ReadData(verbose=True)
    print("------------------------------------------------")

    NN = NeuralNetwork(X_Train.shape[0], [256, 128, 64])
