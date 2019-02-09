#!/usr/bin/env python
# coding: utf-8
# @Author: Kaustav Vats 
# @Roll-Number: 2016048 

#%%
import numpy as np
import matplotlib.pyplot as plt
from math import exp, pi, sqrt, log
import ast
from sklearn.metrics import confusion_matrix
import itertools


def ReadData(filename):
    Arr = []
    Brr = []
    count = 0
    label = 1
    depth = 0
    with open(filename, 'r') as fp:
        line = fp.readlines()
        for row in line:
            # print(row)
            depth += 1
            row = list(map(float, row.split(" ")))
            Arr.append(row)
            # print(row)
            Brr.append(label)
            count += 1
            if count == 10:
                label += 1
                count = 0
    Arr = np.asarray(Arr)
    Brr = np.asarray(Brr)
    return Arr, Brr

def ConfusionMatrix(actual, predicted):
    classes = np.array(['class 1', 'class 2'])
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
    plt.savefig("cnf_matrix.png")

#%%
TrainX ,TrainY = ReadData('bookData.txt')
print(TrainX.shape)
print(TrainY.shape)
# print(TrainY)

def Mean(trainx):
    arr = np.zeros((2, 3), dtype=float)
    for i in range(2):
        count = i*10
        sum = [0, 0, 0]
        for j in range(0, 10):
            # print("I+J" ,count+j)
            sum[0] += trainx[count+j, 0]
            sum[1] += trainx[count+j, 1]
            sum[2] += trainx[count+j, 2]
            print(trainx[count+j, 0])
        sum[0] /= 10
        sum[1] /= 10
        sum[2] /= 10
        arr[i, 0] = sum[0]
        arr[i, 1] = sum[1]
        arr[i, 2] = sum[2]
    # print(arr)
    return arr

# def Mean2(trainx):
#     mean1 = 0
#     for i in range(0, 10):
#         mean1 += trainx[i, 0]
#     mean1 = mean1/10

#     mean2 = 0
#     for i in range(10, 20):
#         mean2 += trainx[i, 0]
#     mean2 = mean2/10
#     return [mean1, mean2]

def Variance(trainx, mean):
    arr = np.zeros((2, 3), dtype=float)
    for i in range(2):
        count = i*10
        sum = [0, 0, 0]
        for j in range(0, 10):
            # print("c", count+j)
            sum[0] += ((trainx[count+j ,0]-mean[i, 0])**2)
            sum[1] += ((trainx[count+j, 1]-mean[i, 1])**2)
            sum[2] += ((trainx[count+j ,2]-mean[i, 2])**2)

        sum[0] = sum[0]/10
        sum[1] = sum[1]/10
        sum[2] = sum[2]/10
        arr[i, 0] = sum[0]
        arr[i, 1] = sum[1]
        arr[i, 2] = sum[2]
    # print(arr)
    return arr

meanData = Mean(TrainX)
print(meanData)
# meanData = Mean2(TrainX)
# print(meanData)
varianceData = Variance(TrainX, meanData)

#%%

def lh(x, mean, variance):
    twosigma = 2*variance
    num = exp(-((x-mean)**2/twosigma))
    den = sqrt(pi*twosigma)
    return num/den

def Testing(trainx, label, mean, variance):
    Correct = 0
    predL = np.zeros((label.shape[0], ), dtype=int)
    print(mean)
    for i in range(20):
        x = trainx[i, 0]
        m = mean[0, 0]
        var = variance[0, 0]
        prob1 = lh(x, m, var)*0.5

        m = mean[1, 0]
        var = variance[1, 0]
        prob2 = lh(x, m, var)*0.5
        print(prob1, prob2, label[i])
        if prob1 > prob2:
            predL[i] = 1
            if label[i] == 1:
                Correct += 1
        else:
            predL[i] = 2
            if label[i] == 2:
                Correct += 1
    print("EE", (20-Correct)*100/20)
    print((Correct/20)*100)
    return predL

predL = Testing(TrainX, TrainY, meanData, varianceData)
# print("Acc:", acc)
ConfusionMatrix(TrainY, predL)


# Bhattacharyya Bound
def BB(pw1, pw2, u1, u2, s1, s2):
    expo = (1/8)*(u2-u1)*(1/((s1+s2)/2))*(u2-u1)+(1/2)*(log(abs((s1+s2)/2)/sqrt(abs(s1)*abs(s2))))
    prod = sqrt(pw1*pw2)
    return prod*exp(-expo)

bbound = BB(0.5, 0.5, meanData[0, 0], meanData[1, 0], varianceData[0, 0], varianceData[1, 0])
print(bbound)
#%%
# Now for two feature values x1, x2

def llmd(x, m, cm):
    # print("X", x)
    # print("m", m)
    # print("cm", cm)

    sub = np.subtract(x, m)
    # print("sub", sub)
    num = exp((-1/2)*(np.matmul(np.matmul(sub, np.linalg.inv(cm)), np.transpose(sub))))
    # print("numnum", num)
    den = 2*pi*sqrt(np.linalg.det(cm))
    return num/den

def findCovMat(trainx, mean):
    mus = [0, 0]
    for i in range(0, 10):
        mus[0] += (trainx[i, 0]-mean[0, 0])*(trainx[i, 1]-mean[0, 1])
        # if i == 0:
            # print(mus)
    mus[0] = mus[0]/10

    for i in range(10, 20):
        mus[1] += (trainx[i, 0]-mean[1, 0])*(trainx[i, 1]-mean[1, 1])
    mus[1] = mus[1]/10
    return mus

CovM = findCovMat(TrainX, meanData)

cov = np.array([
    [varianceData[0, 0], CovM[0]], 
    [CovM[0], varianceData[0, 1]]
])
cov2 = np.array([
    [varianceData[1, 0], CovM[1]],
    [CovM[1], varianceData[1, 1]]
])
u1 = np.array([meanData[0, 0], meanData[0, 1]])
u2 = np.array([meanData[1, 0], meanData[1, 1]])

def Test2Points(trainx, labels, cov, cov2, u1, u2):
    Correct = 0
    PredictL = np.zeros((labels.shape[0], ), dtype=int)
    for i in range(20):
        xy = np.array([trainx[i, 0], trainx[i, 1]])
        prob1 = llmd(xy, u1, cov)*0.5
        prob2 = llmd(xy, u2, cov2)*0.5
        print(prob1, prob2)
        # return None
        if prob1 > prob2:
            PredictL[i] = 1
            if labels[i] == 1:
                Correct += 1
        else:
            PredictL[i] = 2
            if labels[i] == 2:
                Correct += 1
    print("EE", (20-Correct)*100/20)
    print(Correct*100/20)
    return PredictL

predL = Test2Points(TrainX, TrainY, cov, cov2, u1, u2)
print(predL)

ConfusionMatrix(TrainY, predL)

#%%
# Bhattacharyya bound
def BB2(pw1, pw2, u1, u2, cov, cov2):
    sub = np.subtract(u2, u1)
    # print("sub", sub)
    # print("sum", cov+cov2/2)
    covSum = (cov+cov2)/2
    num = (1/8)*np.matmul(np.matmul(sub, np.linalg.inv(covSum)), np.transpose(sub))
    # print("numnum", num)
    num += (1/2)*log(np.linalg.det(covSum)/sqrt(abs(np.linalg.det(cov))*abs(np.linalg.det(cov2))))
    den = sqrt(pw1*pw2)*exp(-num)
    return den
bound = BB2(0.5, 0.5, u1, u2, cov, cov2)
print(bound)


#%%
# For all 3 features x1, x2, x3

def llmd(x, m, cm):
    sub = x-m
    num = exp((-1/2)*(np.matmul(np.matmul(sub, np.linalg.inv(cm)), np.transpose(sub))))
    den = ((2*pi)**(3/2))*sqrt(np.linalg.det(cm))
    return num/den

def findCovMat(trainx, mean):
    mus = np.zeros((3, 3), dtype=float)
    # print(mus)
    for i in range(0, 10):
        mus[0, 1] += (trainx[i, 0]-mean[0, 0])*(trainx[i, 1]-mean[0, 1])
        mus[0, 2] += (trainx[i, 0]-mean[0, 0])*(trainx[i, 2]-mean[0, 2])
        mus[1, 2] += (trainx[i, 1]-mean[0, 1])*(trainx[i, 2]-mean[0, 2])
    mus[0, 1] = mus[0, 1]/10
    mus[0, 2] = mus[0, 2]/10
    mus[1, 2] = mus[1, 2]/10
    mus[1, 0] = mus[0, 1]
    mus[2, 0] = mus[0, 2]
    mus[2, 1] = mus[1, 2]

    mus2 = np.zeros((3, 3), dtype=float)
    for i in range(10, 20):
        mus2[0, 1] += (trainx[i, 0]-mean[1, 0])*(trainx[i, 1]-mean[1, 1])
        mus2[0, 2] += (trainx[i, 0]-mean[1, 0])*(trainx[i, 2]-mean[1, 2])
        mus2[1, 2] += (trainx[i, 1]-mean[1, 1])*(trainx[i, 2]-mean[1, 2])
    mus2[0, 1] = mus2[0, 1]/10
    mus2[0, 2] = mus2[0, 2]/10
    mus2[1, 2] = mus2[1, 2]/10
    mus2[1, 0] = mus2[0, 1]
    mus2[2, 0] = mus2[0, 2]
    mus2[2, 1] = mus2[1, 2]
    return mus, mus2

CovM1, CovM2 = findCovMat(TrainX, meanData)

cov = np.array([
    [varianceData[0, 0], CovM1[0, 1], CovM1[0, 2]], 
    [CovM1[1, 0], varianceData[0, 1], CovM1[1, 2]],
    [CovM1[2, 0], CovM1[2, 1], varianceData[0, 2]]
])

cov2 = np.array([
    [varianceData[1, 0], CovM2[0, 1], CovM2[0, 2]], 
    [CovM2[1, 0], varianceData[1, 1], CovM2[1, 2]],
    [CovM2[2, 0], CovM2[2, 1], varianceData[1, 2]]
])
u1 = np.array([meanData[0, 0], meanData[0, 1], meanData[0, 2]])
u2 = np.array([meanData[1, 0], meanData[1, 1], meanData[1, 2]])

def Test3Points(trainx, labels, cov, cov2, u1, u2):
    Correct = 0
    predL = np.zeros((TrainY.shape[0], ), dtype=int)
    for i in range(20):
        xy = np.array([trainx[i, 0], trainx[i, 1], trainx[i, 2]])
        prob1 = llmd(xy, u1, cov)*0.5
        prob2 = llmd(xy, u2, cov2)*0.5
        print(prob1, prob2)
        # return None
        if prob1 > prob2:
            predL[i] = 1
            if labels[i] == 1:
                Correct += 1
        else:
            predL[i] = 2
            if labels[i] == 2:
                Correct += 1
    print("EE", (20-Correct)*100/20)
    print((Correct*100/20))
    return predL

predL = Test3Points(TrainX, TrainY, cov, cov2, u1, u2)
# print(predL)

ConfusionMatrix(TrainY, predL)

#%%
# Bhattacharyya bound
def BB3(pw1, pw2, u1, u2, cov, cov2):
    sub = np.subtract(u2, u1)
    # print("sub", sub)
    # print("sum", cov+cov2/2)
    covSum = (cov+cov2)/2
    num = (1/8)*np.matmul(np.matmul(sub, np.linalg.inv(covSum)), np.transpose(sub))
    # print("numnum", num)
    num += (1/2)*log(np.linalg.det(covSum)/sqrt(abs(np.linalg.det(cov))*abs(np.linalg.det(cov2))))
    den = sqrt(pw1*pw2)*exp(-num)
    return den
    
bound = BB3(0.5, 0.5, u1, u2, cov, cov2)
print(bound)

#%%
