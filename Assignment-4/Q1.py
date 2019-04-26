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
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve
import itertools

# Neural Netwrok ---------------------------------
class NeuralNetwork:

    def __init__(self, input_size, data_size, output_size, nodes_hidden, verbose=True):
        self.InputSize = input_size
        self.DataSize = data_size
        self.OutputSize = output_size
        self.LayersCount = len(nodes_hidden)
        self.W = []
        self.B = []
        self.Initialize(nodes_hidden)

        if verbose:
            print("------ Neural Network Structure -------")
            for i in range(self.LayersCount):
                print("[W] Layer {} shape: {}".format(i, self.W[i].shape)) 
            print("[W] Layer {} shape: {}".format(self.LayersCount, self.W[self.LayersCount].shape))                    
            for i in range(self.LayersCount):
                print("[B] Layer {} shape: {}".format(i, self.B[i].shape))
            print("[B] Layer {} shape: {}".format(self.LayersCount, self.B[self.LayersCount].shape)) 

    def Initialize(self, nodes_hidden):
        prev = self.DataSize
        for i in range(self.LayersCount):
            now = nodes_hidden[i]
            self.W.append(np.random.randn(prev, now)/np.sqrt(now))
            self.B.append(np.random.randn(1, now)/np.sqrt(now))
            prev = now
        self.W.append(np.random.randn(nodes_hidden[self.LayersCount-1], self.OutputSize))
        self.B.append(np.random.randn(1, self.OutputSize)/np.sqrt(self.OutputSize))        
        # self.W = np.asarray(self.W)
        # self.B = np.asarray(self.B)

    def Convert2OneHotEncoding(self, y_data, verbose=True):
        '''Converts Labels to One-Hot Encoding'''
        result = np.zeros((y_data.shape[0], np.unique(y_data).shape[0]))
        result[np.arange(y_data.shape[0]), y_data] = 1
        if verbose:
            # print(y_data[0], result[0, :])
            print("Converted {} to {}".format(y_data.shape, result.shape))
        return result

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
        Total = np.sum(Exp, axis=1, keepdims=True)
        # print("Exp: {}, Total: {}".format(Exp.shape, Total.shape))
        return Exp / Total

    def FeedForward(self, x_data, actFunc="relu", verbose=True):
        self.A = []
        for i in range(self.LayersCount):
            # print("self.W[{}]: {}".format(i, self.W[i].shape))
            z = None
            if i == 0:
                z = np.dot(x_data, self.W[i]) + self.B[i]
            else:
                z = np.dot(self.A[i-1], self.W[i]) + self.B[i]
            a = self.Sigmoid(z)
            self.A.append(a)
            # print(i)
        # print(self.LayersCount-1)
        z = np.dot(self.A[self.LayersCount-1], self.W[self.LayersCount]) + self.B[self.LayersCount]
        self.SoftMaxOutput = self.Softmax(z)
        # print(z)
        if verbose:
            print("self.SoftMaxOutput: {}".format(self.SoftMaxOutput.shape))

    def BackPropogation(self, x_data, y_data, learning_rate=0.1):
        '''Updating weights of all neurons after training'''
        delta = self.SoftMaxOutput - y_data
        # print(delta.shape)
        dW = (1/self.InputSize) * np.dot(self.A[self.LayersCount-1].T, delta)
        dB = (1/self.InputSize) * np.sum(delta)
        self.W[self.LayersCount] = self.W[self.LayersCount] - learning_rate * dW
        self.B[self.LayersCount] = self.B[self.LayersCount] - learning_rate * dB

        # Deltas = []
        # Deltas.append(delta)
        for i in range(self.LayersCount-1, -1):
            delta = np.dot(delta, self.W[i+1].T) * self.Sigmoid_Derivative(self.A[i])
            if i == 0:
                self.W[i] = self.W[i] - (1/self.DataSize) * learning_rate * np.dot(x_data.T, delta)
            else:    
                self.W[i] = self.W[i] - (1/self.DataSize) * learning_rate * np.dot(self.A[i-1].T, delta)
            self.B[i] = self.B[i] - (1/self.DataSize) * learning_rate * np.sum(delta, axis=0)


    def fit(self, x_data, y_data, batch_size=60000, pretrained=False, epochs=100, learning_rate=0.1, verbose=True):
        if pretrained:
            self.loadModel()
            return
        y_data = self.Convert2OneHotEncoding(y_data, verbose=verbose)
        for i in tqdm(range(epochs)):
            # for j in range(x_data.shape[0]):
            self.FeedForward(x_data, verbose=verbose)
            self.BackPropogation(x_data ,y_data, learning_rate)
            if i % 5 == 0:
                yolo = np.log(self.SoftMaxOutput)
                # print("YOLO: {}".format(yolo))
                loss = np.sum(-y_data * np.log(self.SoftMaxOutput))
                loss /= x_data.shape[0]
                print("\nLoss: {}".format(loss))

    def predict(self, x_test, y_test):
        y_test = self.Convert2OneHotEncoding(y_test)
        A = []
        for i in range(self.LayersCount):
            z = None
            if i == 0:
                z = np.dot(x_test, self.W[i]) + self.B[i]
            else:
                z = np.dot(A[i-1], self.W[i]) + self.B[i]
            a = self.Sigmoid(z)
            A.append(a)
            print(i)
        print("Self.LayerCount: {}".format(self.LayersCount))
        z = np.dot(A[len(A)-1], self.W[self.LayersCount]) + self.B[self.LayersCount]
        SoftMaxOutput = self.Softmax(z)
        return SoftMaxOutput
    
    def score(self, x_test, y_test):
        SoftMaxOutput = self.predict(x_test, y_test)
        Correct = 0
        maxi = np.argmax(SoftMaxOutput, axis=1)

        for i in range(y_test.shape[0]):
            if y_test[i] == maxi[i]:
                Correct += 1
        print("Accuracy: {}%".format(Correct*100/y_test.shape[0]))
        self.ConfusionMatrix(y_test, maxi)

    def ConfusionMatrix(self, actual, predicted, filename="./Data/graph_cm.png"):
            classes = np.unique(predicted)
            # print("classes:", classes)
            cnf_matrix = confusion_matrix(actual, predicted)
            np.set_printoptions(precision=2)
            plt.figure()
            plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = 'd'
            thresh = cnf_matrix.max() / 2.
            for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
                plt.text(j, i, format(cnf_matrix[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cnf_matrix[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            # plt.savefig(filename)
            plt.show()
            plt.close()

    def saveModel(self):
        '''Save Learneed Weigths of the Model'''
        for i in range(len(self.W)):
            np.save("./Data/W{}.npy".format(i), self.W[i])
            np.save("./Data/B{}.npy".format(i), self.B[i])

    def loadModel(self):
        '''Load saved Weights of the Model'''
        self.W = []
        self.B = []
        for i in range(self.LayersCount+1):
            self.W.append(np.load("./Data/W{}.npy".format(i)))
            self.B.append(np.load("./Data/B{}.npy".format(i)))


if __name__ == "__main__":
    '''Reading Mnist Dataset'''
    X_Train, Y_Train, X_Test, Y_Test = ReadData(verbose=True)
    print("------------------------------------------------")
    Scaler = MinMaxScaler()
    X_Train = Scaler.fit_transform(X_Train)
    X_Test = Scaler.fit_transform(X_Test)


    print("------------------------------------------------")
    # Pass Layers including the output layer
    NN = NeuralNetwork(X_Train.shape[0], X_Train.shape[1], 10, [256, 128, 64])
    NN.fit(X_Train, Y_Train, pretrained=True, verbose=False, learning_rate=0.1, epochs=100)
    # NN.saveModel()
    NN.score(X_Train, Y_Train)
    NN.score(X_Test, Y_Test)
