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
            # print(row)
            Arr.append(row)
            # print(row)
    Brr = np.asarray(Arr)
    return Brr

def ReadRisk(filename):
    with open(filename, 'r') as fp:
        line = fp.readline()
        return line

def Read_test_missing(filename):
    Arr = []
    with open(filename, 'r') as fp:
        line = csv.reader(fp)
        for row in line:
            # row = list(map(str, row))
            row[0] = float(row[0])
            if not (row[1] == "NA"):
                row[1] = float(row[1])
            row[2] = int(row[2])
            # print(row)
            Arr.append(row)
            # print(row)
    Brr = np.asarray(Arr)
    return Brr

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

# Mar
Train = ReadData('train.txt')
Test = ReadData('test_all.txt')
TestMissing = Read_test_missing('test_missing.txt')
# print(TestMissing)
# print(TestMissing.dtype)
Risk = ReadRisk('risk.txt')
Risk = ast.literal_eval(Risk)
Visualize(Train)
print(Risk)

for i in range(TestMissing.shape[0]):
    if not TestMissing[i, 1] == "NA":
        TestMissing[i, 1] = float(TestMissing[i, 1])
    TestMissing[i, 0] = float(TestMissing[i, 0])
    TestMissing[i, 2] = int(TestMissing[i, 2])
    
print(TestMissing)

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
        self.CalProb = None

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
        # print(x, m)
        sub = np.subtract(x, m)
        # sub = x-m
        num = exp((-1/2)*(np.matmul(np.matmul(sub, np.linalg.inv(cm)), np.transpose(sub))))
        den = 2*pi*sqrt(abs(np.linalg.det(cm)))
        return num/den

    def Test2Points(self, data, db=True):
        ActualLabel = np.zeros((data.shape[0], ), dtype=int)
        PredictedLabel = np.zeros((data.shape[0], ), dtype=int)
        if db:
            self.CalProb = np.zeros((data.shape[0], 2), dtype=float)
            cou1 = 0
            cou2 = 0
            for i in range(data.shape[0]):
                if data[i, 2] == 0:
                    cou1 += 1
                else:
                    cou2 += 1

        Correct = 0
        for i in range(data.shape[0]):
            prob1 = 0
            prob2 = 0
            if data[i, 1] == "NA":
                y_values = np.random.uniform(-10, 10, 50)
                p1 = 0
                p2 = 0
                for j in y_values:
                    # print("J", j)
                    x_l = float(data[i, 0])
                    x_k = float(j)
                    # print(x_l, x_k)
                    xy = np.array([float(data[i, 0]), float(j)], dtype=float)
                    p1 += self.llmd(xy, self.u1, self.cov)*(self.count0/(self.count0+self.count1))
                    p2 += self.llmd(xy, self.u2, self.cov2)*(self.count1/(self.count0+self.count1))
                prob1 = p1
                prob2 = p2
            else:
                xy = np.array([data[i, 0], data[i, 1]], dtype=float)
                p1 = self.llmd(xy, self.u1, self.cov)*(self.count0/(self.count0+self.count1))
                p2 = self.llmd(xy, self.u2, self.cov2)*(self.count1/(self.count0+self.count1))
                # print(prob1, prob2)
                # return None
                prob1 = p1
                prob2 = p2

            if db:
                self.CalProb[i, 0] = prob1/(prob1+prob2)
                self.CalProb[i, 1] = prob2/(prob1+prob2)
                ActualLabel[i] = data[i, 2]
            if prob1 > prob2:
                PredictedLabel[i] = 0
                if db and int(data[i, 2]) == 0:
                    # print("Hola")
                    Correct += 1
            else:
                PredictedLabel[i] = 1
                if db and int(data[i, 2]) == 1:
                    Correct += 1
            # print(prob1, prob2, ActualLabel[i], PredictedLabel[i])
        if db:
            self.ConfusionMatrix(ActualLabel, PredictedLabel)
            print("EE", ((cou1+cou2)-Correct)*100/(cou1+cou2))
            print(Correct*100/((cou1+cou2)))
        return PredictedLabel

    def ConfusionMatrix(self, actual, predicted):
        classes = np.array(['class 0', 'class 1'])
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

    def RiskTesting(self, data, risk):
        ActualLabel = np.zeros((data.shape[0], ), dtype=int)
        PredictedLabel = np.zeros((data.shape[0], ), dtype=int)

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
            ActualLabel[i] = data[i, 2]
            if prob1 > prob2:
                PredictedLabel[i] = 0
                if data[i, 2] == 0:
                    Correct += 1
            else:
                PredictedLabel[i] = 1
                if data[i, 2] == 1:
                    Correct += 1
        self.ConfusionMatrix(ActualLabel, PredictedLabel)        
        print("EE", (cou1+cou2-Correct)*100/(cou1+cou2))
        return (Correct*100/(cou1+cou2))

    def RocCurve(self, data):
        roc_values = np.zeros((2, 2, 10000))
        for i in range(2):
            Threshold = 1
            for k in range(10000):
                tp = 0
                fn = 0
                fp = 0
                tn = 0
                for j in range(data.shape[0]):
                    classify = False
                    if ( self.CalProb[j, i] > Threshold ):
                        classify = True
                    if classify and data[j, 2] == i:
                        tp += 1
                    elif classify and data[j, 2] != i:
                        fp += 1
                    elif classify == False and data[j, 2] == i:
                        fn += 1
                    elif classify == False and data[j, 2] != i:
                        tn += 1
                roc_values[i, 0, k] = tp/(tp+fn)
                roc_values[i, 1, k] = fp/(tn+fp)

                Threshold = Threshold - Threshold*0.3
            # print("K", k)
        plt.figure()
        color = ['y', 'g']
        plt.plot(roc_values[0, 1, :], roc_values[0, 0, :], color[0], label="ROC Curve for class 0")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc='lower right')
        # plt.show()  
        plt.savefig('roc_c1.png')
        plt.figure()
        plt.plot(roc_values[1, 1, :], roc_values[1, 0, :], color[1], label="ROC Curve for class 1")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc='lower right')
        # plt.show()  
        plt.savefig('roc_c2.png')

    def DecisionBoundary(self, data):
        # ab, ac = plt.subplot(figsize=())
        x = []
        y = []
        x2 = []
        y2 = []
        count = 0
        count2 = 0
        for i in range(len(data)):
            if int(data[i][2]) == 0:
                x.append(float(data[i][0]))
                y.append(float(data[i][1]))
                count += 1
            else:
                x2.append(float(data[i][0]))
                y2.append(float(data[i][1]))
                count2 += 1

        fig = plt.figure()
        plt.scatter(x, y, c='r', label="Class 0 Data Points")
        plt.scatter(x2, y2, c='b', label="Class 1 Data Points")

        x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
        y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1

        # h = (x_max / x_min)/100
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

        NewData = np.c_[xx.ravel(), yy.ravel()]
        print(NewData.shape)
        NewLab = self.Test2Points(NewData, False)
        NewLab = NewLab.reshape(xx.shape)
        # print(xx, xx.shape)
        plt.contour(xx, yy, NewLab, cmap=plt.cm.Paired, alpha=0.8)
        plt.xlabel("X Data Point")
        plt.ylabel("Y Data Point")
        plt.title("Data Visualization - with decision boundary")
        plt.savefig("db.png")

# In[]
mnv = MNV(count0, count1, Train, Test, Risk, meanData, varData)
mnv.findCovMat()

# predLabel = mnv.Test2Points(Train)

# mnv.DecisionBoundary(Train)


# predLabel = mnv.Test2Points(Test)
predLabel = mnv.Test2Points(Test)
mnv.RocCurve(Test)
# mnv.DecisionBoundary(Test)


# mnv.RocCurve(Test)
# In[]
# Using Risk Matrix

for i in range(6):
    risk_matrix = Risk["risk"+str(i+1)]
    print(risk_matrix, type(risk_matrix))
    result = mnv.RiskTesting(Train, risk_matrix)
    print("Acc:", result)
    result = mnv.RiskTesting(Test, risk_matrix)
    print("Test:", result)
# In[]

# Decorrelation Whitening Transform
Visualize(Train)
# https://theclevermachine.wordpress.com/2013/03/30/the-statistical-whitening-transform/
def decorrelation(data):
    # print(data)
    count0 = 0
    count1 = 0
    for i in range(data.shape[0]):
        if data[i, 2] == 0:
            count0 += 1
        else:
            count1 += 1
    print("UO", count0, count1)
    D1, E1 = np.linalg.eigh(mnv.cov)
    D2, E2 = np.linalg.eigh(mnv.cov2)

    print(D1, D1.shape)
    print(E1.shape, E1)

    # cov1
    Dinv = np.diag(D1)
    Dinv = np.linalg.inv(np.sqrt(Dinv))
    # Dinv = np.diag(1. / np.sqrt(D1))
    
    ETrans = np.transpose(E1)
    Einv = np.linalg.inv(E1)
    W1 = np.matmul(np.matmul(Einv, Dinv), ETrans)
    print(W1)
    # cov2  
    Dinv = np.diag(D2)
    Dinv = np.linalg.inv(np.sqrt(Dinv))
    # Dinv = np.diag(1. / np.sqrt(D2))

    ETrans = np.transpose(E2)
    Einv = np.linalg.inv(E2)
    W2 = np.matmul(np.matmul(Einv, Dinv), ETrans)
    print(W2)

    NewTrain0 = np.zeros((2, count0), dtype=float)
    NewTrain1 = np.zeros((2, count1), dtype=float)

    counter1 = 0
    counter2 = 0
    for i in range(data.shape[0]):
        if data[i, 2] == 0:
            print("NO")
            NewTrain0[0, counter1] = data[i, 0]
            NewTrain0[1, counter1] = data[i, 1]
            counter1 += 1
        else:
            print(counter2, i)
            NewTrain1[0, counter2] = data[i, 0]
            NewTrain1[1, counter2] = data[i, 1]
            counter2 += 1

    NewTrain0 = np.matmul(W1, NewTrain0)
    NewTrain1 = np.matmul(W2, NewTrain1)

    NTrain = np.zeros((data.shape[0], 3) , dtype=float)
    index = 0
    for i in range(count0):
        NTrain[index, 0] = NewTrain0[0, i]
        NTrain[index, 1] = NewTrain0[1, i]
        NTrain[index, 2] = 0
        index += 1
    for i in range(count1):
        NTrain[index, 0] = NewTrain1[0, i]
        NTrain[index, 1] = NewTrain1[1, i]
        NTrain[index, 2] = 1
        index += 1
    print(index, count0+count1)
    
    return NTrain

NTrain = decorrelation(Train)
# print("test", type(Test[0, 2]))
NTest = decorrelation(Test)
Visualize(NTrain)
# Visualize(NTest)



# In[]

meanData = Mean(NTrain, count0, count1)
varData = Variance(NTrain, meanData)
mnv = MNV(count0, count1, NTrain, NTest, Risk, meanData, varData)
mnv.findCovMat()

predLabel = mnv.Test2Points(NTrain)
# mnv.RocCurve(NTrain)

# result, predLabel = mnv.Test2Points(NTest)
# print(result)
# mnv.RocCurve(Test)

mnv.DecisionBoundary(NTrain)


# In[]
# Using Risk Matrix

for i in range(6):
    risk_matrix = Risk["risk"+str(i+1)]
    print(risk_matrix, type(risk_matrix))
    result = mnv.RiskTesting(NTrain, risk_matrix)
    print("Acc:", result)
    result = mnv.RiskTesting(NTest, risk_matrix)
    print("Test:", result)


#%%
