# @Author: Kaustav Vats 
# @Roll-Number: 2016048 

# In[]
import numpy as np
import csv
import matplotlib.pyplot as plt
from math import exp, pi, sqrt
import ast
from sklearn.metrics import confusion_matrix
import itertools


def ReadData(filename):
    Arr = []
    with open(filename, 'r') as fp:
        line = csv.reader(fp)
        for row in line:
            row = list(map(float, row))
            row[2] = int(row[2])
            Arr.append(row)
            # print(row)
    Brr = np.asarray(Arr)
    return Brr

def ReadRisk(filename):
    with open(filename, 'r') as fp:
        line = fp.readline()
        return line

def Visualize(train):
    x = []
    y = []
    x2 = []
    y2 = []
    count = 0
    count2 = 0
    color = ['r', 'b']
    for i in range(len(train)):
        if int(train[i][2]) == 0:
            x.append(float(train[i][0]))
            y.append(float(train[i][1]))
            count += 1
        else:
            x2.append(float(train[i][0]))
            y2.append(float(train[i][1]))
            count2 += 1
    print("count", count)
    print("count2", count2)

    plt.figure()
    plt.plot(x, y, 'o', color=color[0], label="Class 0 Data Points")
    plt.plot(x2, y2, 'o', color=color[1], label="Class 1 Data Points")
    plt.xlabel("X Data Point")
    plt.ylabel("Y Data Point")
    plt.title("Data Visualization")
    plt.legend(loc='lower right')
    # plt.show()  
    plt.savefig('visualize.png')
    
# In[]
Train = ReadData('train.txt')
Test = ReadData('test_all.txt')
Risk = ReadRisk('risk.txt')
Risk = ast.literal_eval(Risk)
Visualize(Train)
print(Risk)

# In[]
count0 = 0
count1 = 0

for i in range(Train.shape[0]):
    if Train[i, 2] == 0:
        count0 += 1
    else:
        count1 += 1

print(count0, count1)
def Mean(trainxy, c1, c2):
    mean = np.zeros((2, 2), dtype=float)

    for i in range(trainxy.shape[0]):
        if trainxy[i, 2] == 0:
            mean[0, 0] += trainxy[i, 0]
            mean[0, 1] += trainxy[i, 1]
            # count0 += 1
        else:
            mean[1, 0] += trainxy[i, 0]
            mean[1, 1] += trainxy[i, 1]
            # count1 += 1
    mean[0, 0] = mean[0, 0]/c1
    mean[0, 1] = mean[0, 1]/c1
    mean[1, 0] = mean[1, 0]/c2
    mean[1, 1] = mean[1, 1]/c2
    return mean

def Variance(trainxy, mean):
    count0 = 0
    count1 = 0
    var = np.zeros((2, 2), dtype=float)

    for i in range(trainxy.shape[0]):
        if trainxy[i, 2] == 0:
            var[0, 0] += (trainxy[i, 0]-mean[0, 0])**2
            var[0, 1] += (trainxy[i, 1]-mean[0, 1])**2
            count0 += 1
        else:
            var[1, 0] += (trainxy[i, 0]-mean[1, 0])**2
            var[1, 1] += (trainxy[i, 1]-mean[1, 1])**2
            count1 += 1
    var[0, 0] = var[0, 0]/count0
    var[0, 1] = var[0, 1]/count0
    var[1, 0] = var[1, 0]/count1
    var[1, 1] = var[1, 1]/count1
    return var

meanData = Mean(Train, count0, count1)
varData = Variance(Train, meanData)
# print(meanData.shape)
# print(meanData)
# print(varData)

class MNV:

    def __init__(self, c1, c2, train, test, risk, meanData, varData):
        self.count0 = c1
        self.count1 = c2
        self.trainx = train
        self.testx = test
        self.meanData = meanData
        self.varData = varData
        self.u1 = None
        self.u2 = None
        self.cov = None
        self.cov2 = None

    def findCovMat(self):
        mus = [0, 0]
        count = [0, 0]
        for i in range(self.trainx.shape[0]):
            if self.trainx[i, 2] == 0:
                count[0] += 1
                mus[0] += (self.trainx[i, 0]-self.meanData[0, 0])*(self.trainx[i, 1]-self.meanData[0, 1])
            else:
                count[1] += 1
                mus[1] += (self.trainx[i, 0]-self.meanData[1, 0])*(self.trainx[i, 1]-self.meanData[1, 1])
        mus[0] = mus[0]/count[0]
        mus[1] = mus[1]/count[1]
        self.cov = np.array([
            [varData[0, 0], mus[0]], 
            [mus[0], varData[0, 1]]
        ])
        self.cov2 = np.array([
            [varData[1, 0], mus[1]],
            [mus[1], varData[1, 1]]
        ])
        self.u1 = np.array([meanData[0, 0], meanData[0, 1]])
        self.u2 = np.array([meanData[1, 0], meanData[1, 1]])
        # return mus

    def llmd(self, x, m, cm):
        # sub = np.subtract(x, m)
        sub = x-m
        num = exp((-1/2)*(np.matmul(np.matmul(sub, np.linalg.inv(cm)), np.transpose(sub))))
        den = 2*pi*sqrt(abs(np.linalg.det(cm)))
        return num/den

    def Test2Points(self, data):
        cou1 = 0
        cou2 = 0
        for i in range(data.shape[0]):
            if data[i, 2] == 0:
                cou1 += 1
            else:
                cou2 += 1

        Correct = 0
        for i in range(data.shape[0]):
            xy = np.array([data[i, 0], data[i, 1]])
            prob1 = self.llmd(xy, self.u1, self.cov)*(self.count0/(self.count0+self.count1))
            prob2 = self.llmd(xy, self.u2, self.cov2)*(self.count1/(self.count0+self.count1))
            # print(prob1, prob2)
            # return None
            if prob1 > prob2:
                if data[i, 2] == 0:
                    Correct += 1
            else:
                if data[i, 2] == 1:
                    Correct += 1
        print("EE", ((cou1+cou2)-Correct)*100/(cou1+cou2))
        return (Correct*100/((cou1+cou2)))

    def RiskTesting(self, data, risk):
        cou1 = 0
        cou2 = 0
        for i in range(data.shape[0]):
            if data[i, 2] == 0:
                cou1 += 1
            else:
                cou2 += 1

        Correct = 0
        for i in range(data.shape[0]):
            xy = np.array([data[i, 0], data[i, 1]])
            prob1 = self.llmd(xy, self.u1, self.cov)*(self.count0/(self.count0+self.count1))
            prob2 = self.llmd(xy, self.u2, self.cov2)*(self.count1/(self.count0+self.count1))
            prob1 = prob1*(risk[1][0]-risk[0][0])
            prob2 = prob2*(risk[0][1]-risk[1][1])

            # print(prob1, prob2)
            # return None
            if prob1 > prob2:
                if data[i, 2] == 0:
                    Correct += 1
            else:
                if data[i, 2] == 1:
                    Correct += 1
        # print("EE", (cou1+cou2-Correct)*100/(cou1+cou2))
        return (Correct*100/(cou1+cou2))

mnv = MNV(count0, count1, Train, Test, Risk, meanData, varData)
mnv.findCovMat()

# CovM = findCovMat(Train, meanData)

# cov = np.array([
#     [varData[0, 0], CovM[0]], 
#     [CovM[0], varData[0, 1]]
# ])
# cov2 = np.array([
#     [varData[1, 0], CovM[1]],
#     [CovM[1], varData[1, 1]]
# ])
# u1 = np.array([meanData[0, 0], meanData[0, 1]])
# u2 = np.array([meanData[1, 0], meanData[1, 1]])

# print(u1, u2)
# print(cov, cov2)
# def llmd(x, m, cm):
#     # sub = np.subtract(x, m)
#     sub = x-m
#     num = exp((-1/2)*(np.matmul(np.matmul(sub, np.linalg.inv(cm)), np.transpose(sub))))
#     den = 2*pi*sqrt(abs(np.linalg.det(cm)))
#     return num/den

# def Test2Points(trainx, cov, cov2, u1, u2):
#     cou1 = 0
#     cou2 = 0
#     for i in range(trainx.shape[0]):
#         if trainx[i, 2] == 0:
#             cou1 += 1
#         else:
#             cou2 += 1

#     Correct = 0
#     for i in range(trainx.shape[0]):
#         xy = np.array([trainx[i, 0], trainx[i, 1]])
#         prob1 = llmd(xy, u1, cov)*(count0/(count0+count1))
#         prob2 = llmd(xy, u2, cov2)*(count1/(count0+count1))
#         # print(prob1, prob2)
#         # return None
#         if prob1 > prob2:
#             if trainx[i, 2] == 0:
#                 Correct += 1
#         else:
#             if trainx[i, 2] == 1:
#                 Correct += 1
#     print("EE", ((cou1+cou2)-Correct)*100/(cou1+cou2))
#     return (Correct*100/((cou1+cou2)))

result = mnv.Test2Points(Train)
print(result)

result = mnv.Test2Points(Test)
print(result)
# In[]
# Using Risk Matrix

# def RiskTesting(trainx, cov, cov2, u1, u2, risk):
#     cou1 = 0
#     cou2 = 0
#     for i in range(trainx.shape[0]):
#         if trainx[i, 2] == 0:
#             cou1 += 1
#         else:
#             cou2 += 1

#     Correct = 0
#     for i in range(trainx.shape[0]):
#         xy = np.array([trainx[i, 0], trainx[i, 1]])
#         prob1 = llmd(xy, u1, cov)*(count0/(count0+count1))
#         prob2 = llmd(xy, u2, cov2)*(count1/(count0+count1))
#         prob1 = prob1*(risk[1][0]-risk[0][0])
#         prob2 = prob2*(risk[0][1]-risk[1][1])

#         # print(prob1, prob2)
#         # return None
#         if prob1 > prob2:
#             if trainx[i, 2] == 0:
#                 Correct += 1
#         else:
#             if trainx[i, 2] == 1:
#                 Correct += 1
#     # print("EE", (cou1+cou2-Correct)*100/(cou1+cou2))
#     return (Correct*100/(cou1+cou2))

for i in range(5):
    risk_matrix = Risk["risk"+str(i+1)]
    print(risk_matrix, type(risk_matrix))
    result = mnv.RiskTesting(Train, risk_matrix)
    print("Acc:", result)
    result = mnv.RiskTesting(Test, risk_matrix)
    print("Test:", result)
# In[]



