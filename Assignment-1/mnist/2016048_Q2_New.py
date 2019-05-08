#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Kaustav Vats (2016048)
from __future__ import division
import numpy as np
from mnist.utils import mnist_reader
import pathlib
from math import log
import matplotlib.pyplot as plt

# In[2]:
X_train, Y_train = mnist_reader.load_mnist('mnist/data', kind='train')
X_test, Y_test = mnist_reader.load_mnist('mnist/data', kind='t10k')

def Binarize(arr):
    NewArr = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.int)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] <= 127:
                NewArr[i, j] = 0
            else:
                NewArr[i, j] = 1
    return NewArr

if pathlib.Path('xtrain.npy').exists():
    X_train = np.load('xtrain.npy')
else:
    X_train = Binarize(X_train)
    np.save('xtrain.npy', X_train)

if pathlib.Path('xtest.npy').exists():
    X_test = np.load('xtest.npy')
else:
    X_test = Binarize(X_test)
    np.save('xtest.npy', X_test)

c1 = 1
c2 = 8
count0 = 0
count1 = 0
for i in range(60000):
    if Y_train[i] == c1:
        count0 += 1
    if Y_train[i] == c2:
        count1 += 1
print(count0, count1, count0+count1)

X_train_New = np.zeros((count0+count1, 784), dtype=np.int)
Y_train_New = np.zeros((count0+count1, ), dtype=np.int)

k=0
for i in range(X_train.shape[0]):
    if Y_train[i] == c1 or Y_train[i] == c2:
        for j in range(X_train.shape[1]):
            X_train_New[k, j] = X_train[i, j]
        if Y_train[i] == c1:
            Y_train_New[k] = 0
        else:
            Y_train_New[k] = 1
        k += 1

count2 = 0
count3 = 0
for i in range(10000):
    if Y_test[i] == c1:
        count2 += 1
    if Y_test[i] == c2:
        count3 += 1
print(count2, count3, count2+count3)

X_test_New = np.zeros((count2+count3, 784), dtype=np.int)
Y_test_New = np.zeros((count2+count3, ), dtype=np.int)

k=0
for i in range(X_test.shape[0]):
    if Y_test[i] == c1 or Y_test[i] == c2:
        for j in range(X_test.shape[1]):
            X_test_New[k, j] = X_test[i, j]
        if Y_test[i] == c2:
            Y_test_New[k] = 0
        else:
            Y_test_New[k] = 1
        k += 1

# In[3]:

# Naive Bayes Implementation

class NaiveBayes:
    def __init__(self, ClassCount, testCount):
        self.ClassCount = ClassCount
        self.Prob = np.zeros((2, 784, 2))
        self.Pred = np.zeros(2)
        self.PredictedClass = np.zeros((testCount,), dtype=np.int)
        self.ConfusionMatrix = np.zeros((2, 2), dtype=np.int)
        self.CalProb = np.zeros((testCount, 10))
        
    def fit(self, x_train, y_train, c1Count, c2Count):
        if pathlib.Path('Prob12.npy').exists():
            self.Prob = np.load('Prob12.npy')
        else:
            print(x_train.shape[0], "x", x_train.shape[1])
            for i in range(0, x_train.shape[0]):
                for j in range(0, x_train.shape[1]):
                    if x_train[i, j] == 0:
                        self.Prob[y_train[i], j, 0] += 1
                    else:
                        self.Prob[y_train[i], j, 1] += 1
            for j in range(x_train.shape[1]):
                self.Prob[0, j, 0] = self.Prob[0, j, 0]/c1Count
                self.Prob[0, j, 1] = self.Prob[0, j, 1]/c1Count
                self.Prob[1, j, 0] = self.Prob[1, j, 0]/c2Count
                self.Prob[1, j, 1] = self.Prob[1, j, 1]/c2Count
            # print(self.Prob[0, :, 0])
            np.save('Prob12.npy', self.Prob)

    def predict(self, x_test, y_test, ImgCount):
        if pathlib.Path('CalProb.npy').exists():
            self.CalProb = np.load('CalProb.npy')
        else:
            Correct = 0
            print(x_test.shape[0], "x", x_test.shape[1])
            for i in range(x_test.shape[0]):
                self.Pred = np.zeros(2)
                for j in range(self.ClassCount):
                    num = 1.0
                    for k in range(x_test.shape[1]):
                        num = num*self.Prob[j, k, x_test[i, k]]
                    num = num*0.5
                    self.Pred[j] = num
                print(self.Pred)

                if self.Pred[0] > self.Pred[1]:
                    self.PredictedClass[i] = 0
                else:
                    self.PredictedClass[i] = 1

                if self.PredictedClass[i] == y_test[i]:
                    Correct += 1

                self.Pred = self.Pred/(np.sum(self.Pred))

                for n in range(self.ClassCount):
                    self.CalProb[i, n] = self.Pred[n]
            print(Correct/ImgCount)
            print(self.PredictedClass[:])
            print(y_test[:])
            np.save('CalProb.npy', self.CalProb)

    def ConMatrix(self, y_test):
        for i in range(y_test.shape[0]):
            self.ConfusionMatrix[y_test[i], self.PredictedClass[i]] += 1
        print(self.ConfusionMatrix)

    def PrecisionAndRecall(self, ImgCount):
        for i in range(self.ClassCount):
            tpfp = 0.0
            for j in range(self.ClassCount):
                tpfp += self.ConfusionMatrix[j, i]
            print("Precision for Class", i, "=", self.ConfusionMatrix[i, i]/tpfp)
            print("Recall for Class", i, "=", self.ConfusionMatrix[i, i]/ImgCount)
            print()

    def RocCurve(self, y_test):
        # print(max(self.Prob[5, :, 0]))
        roc_values = np.zeros((2, 2, 1000))
        if pathlib.Path('roc_value.npy').exists():
            roc_values = np.load('roc_value.npy')
        else:
            for i in range(2):
                Threshold = 1.0
                for k in range(1000):
                    tp = 0
                    fn = 0
                    fp = 0
                    tn = 0
                    for j in range(y_test.shape[0]):
                        classify = False
                        if ( self.CalProb[j, i] > Threshold ):
                            classify = True
                        if classify and y_test[j] == i:
                            tp += 1
                        elif classify and y_test[j] != i:
                            fp += 1
                        elif classify == False and y_test[j] == i:
                            fn += 1
                        else:
                            tn += 1

                    roc_values[i, 0, k] = tp/(tp+fn)
                    roc_values[i, 1, k] = fp/(tn+fp)

                    Threshold *= 0.3
                print("K", k)
            np.save('roc_value.npy', roc_values)

        plt.figure()
        color = ['b', 'g']
        for i in range(2):
            plt.plot(roc_values[i, 1, :], roc_values[i, 0, :], color[i], label="ROC Curve for class %d " %i)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc='lower right')
        # plt.show()  
        plt.savefig('roc_q2.png')

# In[4]:

NB = NaiveBayes(2, count2+count3)
NB.fit(X_train_New, Y_train_New, count0, count1)

# In[5]:
NB.predict(X_test_New, Y_test_New, count2+count3)
NB.ConMatrix(Y_test_New)

# In[6]:
NB.PrecisionAndRecall(count2+count3)
NB.RocCurve(Y_test_New)


#%%
