# Kaustav Vats (2016048)

import numpy as np
from glob import glob
import cv2
from numpy import linalg as la
from sklearn.naive_bayes import GaussianNB
from math import ceil
from random import randint as randi
from sklearn.tree import DecisionTreeClassifier
import csv
import matplotlib.pyplot as plt

def ReadData1():
    imageList = []
    imageLabels = []
    for i in range(1, 12):
        folderImages = glob(".\\Q1_dataset\\Face_data\\"+str(i)+"\\*")
        for item in folderImages:
            img = cv2.imread(item, 0)
            img = cv2.resize(img, (50, 50))
            imageList.append(img.flatten())
            imageLabels.append(i)
        # print(np.array(imageList).shape)

    imageList = np.array(imageList)
    # print(imageList.shape)
    imageLabels = np.asarray(imageLabels)
    return imageList, imageLabels

def ReadDataQ2():
    data = []
    label = []
    with open('Q2_dataset/letter-recognition.data', 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            # print(type(line[0]), ord(line[0]))
            label.append(ord(line.pop(0))-65)
            line = np.asarray(line)
            data.append(line)
    label = np.asarray(label)
    data = np.asarray(data, dtype=np.float64)
    file.close()
    return data, label

def SplitData(X_Data, Y_Data, amount):
    DataSize = X_Data.shape[0]
    Testing_Size = ceil(DataSize * amount/100)
    DataSlot = []
    # print (Testing_Size, DataSize)
    for i in range(DataSize):
        DataSlot.append(i)
    TrainingData = []
    TrainingLabel = []
    TestingData = []
    TestingLabels = []
    while(Testing_Size > 0):
        index = randi(0, len(DataSlot)-1)
        # print("index:", index, len(DataSlot))
        val = DataSlot.pop(index)
        TrainingData.append(X_Data[val])
        TrainingLabel.append(Y_Data[val])
        Testing_Size -= 1
    # print(len(DataSlot))
    size = len(DataSlot)
    for i in range(size):
        index = DataSlot.pop()
        TestingData.append(X_Data[index])
        TestingLabels.append(Y_Data[index])
    print("NewSizes:", len(TrainingData), len(TestingData))
    return np.asarray(TrainingData), np.asarray(TrainingLabel), np.asarray(TestingData), np.asarray(TestingLabels)

def GaussianClassifier(x_train, y_train, x_test, y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)

    Predicted = clf.predict(x_train)
    count1 = 0
    print(Predicted.shape)
    for i in range(len(y_train)):
        if (y_train[i] == Predicted[i]):
            count1 += 1
    print("Accuracy[Train]:", (count1/len(y_train))*100)

    Predicted = clf.predict(x_test)
    count2 = 0
    # print(Predicted.shape)
    for i in range(len(y_test)):
        if (y_test[i] == Predicted[i]):
            count2 += 1
    print("Accuracy[Test]:", (count2/len(y_test))*100) 

    return (count2/len(y_test))*100

def DTClassifier(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=5)
    clf.fit(x_train, y_train)

    Predicted = clf.predict(x_train)
    count1 = 0
    print(Predicted.shape)
    for i in range(len(y_train)):
        if (y_train[i] == Predicted[i]):
            count1 += 1
    print("Accuracy[Train]:", (count1/len(y_train))*100)

    Predicted = clf.predict(x_test)
    count2 = 0
    # print(Predicted.shape)
    for i in range(len(y_test)):
        if (y_test[i] == Predicted[i]):
            count2 += 1
    print("Accuracy[Test]:", (count2/len(y_test))*100) 

    return (count2/len(y_test))*100

def PCA(x_data, x_test, ee):
    # Data Normalization
    x_data = NormalizeData(x_data)
    # print("NEW DATA\n", NX_Train)

    # Eigen Value Decomposition
    EigenVal, EigenVec = EigenValueDecomposition(x_data)
    # print("EigenVal:", EigenVal.shape)
    # print("EigenVec:", EigenVec.shape)

    ProjectionMatrix = EigenValueProjection(EigenVal, EigenVec, ee)
    # print (ProjectionMatrix)
    # NX_Train = ProjectedData(X_Train, ProjectionMatrix)
    # NX_Test = ProjectedData(X_Test, ProjectionMatrix)
    x_data = ProjectedData(x_data, ProjectionMatrix)
    x_test = ProjectedData(x_test, ProjectionMatrix)
    return x_data, x_test

def NormalizeData(data):
    mean = np.mean(data, axis=0, keepdims=True)
    # print(mean)
    new_data = data - mean
    return new_data


def EigenValueDecomposition(data):
    CovarienceMatrix = np.cov(data.T)
    # CovarienceMatrix = CovarienceMatrix/(data.shape[0]-1)
    EigenValues, EigenVector = la.eig(CovarienceMatrix)
    EigenValues = np.abs(EigenValues)
    EigenVector = np.real(EigenVector)
    # print (EigenVector)
    return EigenValues, EigenVector

def EigenValueProjection(eigen_val, eigen_vector, ee):
    eigen_pair = []
    TotalEigenVal = 0
    for i in range(len(eigen_val)):
        eigen_pair.append((eigen_val[i], eigen_vector[:, i]))
        TotalEigenVal += eigen_val[i]
    eigen_pair.sort(key=lambda k: k[0], reverse=True)
    # print(eigen_pair[0][1].shape)
    total_ee = 0
    W = []
    for i in range(len(eigen_val)):
        # print(eigen_pair[i][0])
        # print((total_ee + eigen_pair[i][0])*100/TotalEigenVal)
        if ((total_ee + eigen_pair[i][0])*100/TotalEigenVal < ee):
            total_ee += eigen_pair[i][0]
            W.append(eigen_pair[i][1])
        else:
            total_ee += eigen_pair[i][0]
            W.append(eigen_pair[i][1])
            break
    # print(W[0].shape)
    W = np.asarray(W)
    # print("Printing Projection Matrix")
    # print(W.shape)
    # print(W[0])
    return W

def ProjectedData(x_data, projected_matrix):
    result = np.matmul(x_data, projected_matrix.T)
    return result

# http://goelhardik.github.io/2016/10/04/fishers-lda/
def LDA(x_data, y_data, x_test):
    Sb = BetweenClassScatter(x_data, y_data, [1, 12])
    Sw = WithinClassScatter(x_data, y_data, [1, 12])

    M = np.matmul(la.pinv(Sw), Sb)
    # CovarienceMatrix = np.cov(M.T)
    EigenValues, EigenVector = la.eig(M)
    EigenValues = np.abs(EigenValues)
    EigenVector = np.real(EigenVector)
    # eigen_pair = []
    # for i in range(len(EigenValues)):
    #     eigen_pair.append((EigenValues[i], EigenVector[:, i]))
    # eigen_pair.sort(key=lambda k: k[0], reverse=True)
    
    # W = []
    # for i in range(len(eigen_pair)):
    #     W.append(eigen_pair[i][1])
    W = EigenVector
    # W = np.asarray(W)
    print("Original Shape:", x_data.shape)
    x_data = ProjectedData(x_data, W)
    print("Projected Shape:", x_data.shape)
    x_test = ProjectedData(x_test, W)
    return x_data, x_test

def GetClassData(x_data, y_data, w):
    NewData = []
    for j in range(x_data.shape[0]):
        if ( y_data[j] == w ):
            NewData.append(x_data[j])
    NewData = np.asarray(NewData)
    return NewData

def BetweenClassScatter(x_data, y_data, classRange):
    mean_value = np.mean(x_data, axis=0)
    finalMatrix = np.zeros((x_data.shape[1], x_data.shape[1]))
    for i in range(classRange[0], classRange[1]):
        NewData = GetClassData(x_data, y_data, i)
        MeanData = np.mean(NewData, axis=0)
        # print(str(i), MeanData)
        result = MeanData - mean_value
        result = np.reshape(result, (result.shape[0], 1))
        # result = np.reshape(result, (1, result.shape[0]))
        # print(result)
        # print("result:", result.shape)
        result = result * NewData.shape[0]
        result = np.matmul(result, result.T)
        # print(result)
        # print("result:", result.shape)
        finalMatrix += result
    print("Sb:", finalMatrix.shape)
    # print(finalMatrix)
    return finalMatrix

def WithinClassScatter(x_data, y_data, classRange):
    finalMatrix = np.zeros((x_data.shape[1], x_data.shape[1]))
    for i in range(classRange[0], classRange[1]):
        NewData = GetClassData(x_data, y_data, i)
        # print("NewData, Sw:", NewData.shape)
        mean_class = np.mean(NewData, axis=0)
        result = NewData - mean_class
        result = np.matmul(result.T, result)
        finalMatrix += result
    print("Sw:", finalMatrix.shape)
    # print(finalMatrix)
    return finalMatrix


def NFold(N, x_train, y_train, x_test, y_test):
    indexArr = np.arange(x_train.shape[0])
    np.random.shuffle(indexArr)
    # print(indexArr)
    AllBins = []
    Bin = []
    LabelBin = []
    AllLabelBins = []
    count = 0
    tempindex = 0
    i = 0
    while(i < x_train.shape[0]):
        if (count < x_train.shape[0]//N or tempindex == N-1):
            Bin.append(x_train[indexArr[i]])
            LabelBin.append(y_train[indexArr[i]])
            count += 1
        else:
            count = 0
            tempindex += 1
            AllBins.append(np.asarray(Bin))
            AllLabelBins.append(np.asarray(LabelBin))
            Bin = []
            LabelBin = []
            i -= 1
        i += 1
    AllBins.append(np.asarray(Bin))
    AllLabelBins.append(np.asarray(LabelBin))

    return AllBins, AllLabelBins


def getAlpha(error):
    # res = 0.5 * np.log((1-error)/error) + np.log(25)
    res = np.log((1-error)/error) + np.log(25)
    return res

def AdaBoost(n, x_train, y_train, x_test, y_test):
    w = np.ones(x_train.shape[0], dtype=np.float) / x_train.shape[0]
    clf = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=5)
    TrainPredicted = np.zeros((x_train.shape[0], 26), dtype=np.float64)
    TestPredicted = np.zeros((x_test.shape[0], 26), dtype=np.float64)
    ErrorListTrain = []
    ErrorListTest = []

    for i in range(n):
        # print("Weights", w)
        clf.fit(x_train, y_train, sample_weight=w)
        predicted_train = clf.predict(x_train)
        Train_Probs = clf.predict_proba(x_train)
        predicted_test = clf.predict(x_test)
        Test_Probs = clf.predict_proba(x_test)


        miss = []
        error = 0
        for j in range(x_train.shape[0]):
            if (predicted_train[j] != y_train[j]):
                error += w[j]
                miss.append(1)
            else:
                miss.append(0)

        error /= np.sum(w)
        if (error == 0):
            print(w)
            print(predicted_train)
            print(y_train)
            print(error, i)
            break
        # print("Error:", error)

        alpha = getAlpha(error)
        # print("alpha:", alpha)
        for j in range(len(miss)):
            w[j] = w[j] * np.exp(miss[j]*alpha)

        w /= np.sum(w)
        
        TrainPredicted += alpha * Train_Probs
        TestPredicted += alpha * Test_Probs

        predicted_labels_train = np.argmax(TrainPredicted, axis=1)
        predicted_labels_test = np.argmax(TestPredicted, axis=1)
        Acc1, err1 = GetResults(predicted_labels_train, y_train)
        Acc2, err2 = GetResults(predicted_labels_test, y_test)
        ErrorListTrain.append(err1)
        ErrorListTest.append(err2)

    predicted_labels_train = np.argmax(TrainPredicted, axis=1)
    predicted_labels_test = np.argmax(TestPredicted, axis=1)
    Acc1, err1 = GetResults(predicted_labels_train, y_train)
    Acc2, err2 = GetResults(predicted_labels_test, y_test)
    
    plotError(ErrorListTrain, np.arange(len(ErrorListTrain)), ErrorListTest)
    print("[AdaBoost Train] Accuracy:", Acc1)
    print("[AdaBoost Test] Accuracy:", Acc2)

def plotError(x_train, y, x_test):
    plt.figure()
    plt.plot(y, x_train, label="Training Error Curve")
    plt.plot(y, x_test, label="Testing Error Curve")
    plt.ylabel("Error Rate")
    plt.xlabel("Iterations")
    plt.title("Error Rate vs Iterations")
    plt.legend(loc='upper right')
    plt.savefig("Q2_Data/AdaBoost_ErrorCurve.png")

def GetResults(PredictedLabels, ActualLabels):
    correct = 0
    error = 0
    for i in range(PredictedLabels.shape[0]):
        if (PredictedLabels[i] == ActualLabels[i]):
            correct += 1
        else:
            error += 1
    return correct*100/PredictedLabels.shape[0], error/PredictedLabels.shape[0]

def Bagging(n, x_train, y_train, x_test, y_test):
    # print(x_train.shape)
    # DataMemory = []
    BagsX = []
    BagsY = []
    BagSize = x_train.shape[0]
    for i in range(n):
        # SmallDataMemory = []
        SmallBagX = []
        SmallBagY = []
        for j in range(BagSize):
            index = randi(0, x_train.shape[0]-1)
            # SmallDataMemory.append(index)
            SmallBagX.append(x_train[index])
            SmallBagY.append(y_train[index])
        # DataMemory.append(SmallDataMemory)
        SmallBagX = np.asarray(SmallBagX)
        SmallBagY = np.asarray(SmallBagY)
        BagsX.append(SmallBagX)
        BagsY.append(SmallBagY)
    BagsX = np.asarray(BagsX)
    BagsY = np.asarray(BagsY)   
    # print(BagsX.shape) 

    # TrainPredicted = np.zeros((x_train.shape[0], 26), dtype=np.int)
    TestPredicted = np.zeros((x_test.shape[0], 26), dtype=np.int)
    # ErrorListTrain = []
    # ErrorListTest = []
    clf = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=5)
    for i in range(n):
        # print(BagsX[i])
        # print("Bag"+str(i))
        # print(BagsX[i].shape)
        # print(BagsY[i].shape)
        clf.fit(BagsX[i], BagsY[i])

        # Pred_Train = clf.predict(BagsX[i])
        # Acc_train = clf.score(BagsX[i], BagsY[i])
        # Train_Probs = clf.predict_proba(BagsX[i])

        Pred_Test = clf.predict(x_test)
        # Acc_test = clf.score(x_test, y_test)
        # Test_Probs = clf.predict_proba(x_test)

        # TrainPredicted += Train_Probs
        # TestPredicted += Test_Probs

        # print("[Bagging Train] Accuracy:" + str(i), Acc_train)
        # print("[Bagging Test] Accuracy:" + str(i), Acc_test)
        # for k in range(len(DataMemory[i])):
        #     # for j in range(Pred_Train.shape[0]):
        #     index = DataMemory[i][k]
        #     TrainPredicted[index, Pred_Train[k]] += 1
        
        for j in range(x_test.shape[0]):
            TestPredicted[j, Pred_Test[j]] += 1

        # predicted_labels_train = np.argmax(TrainPredicted, axis=1)
        # predicted_labels_test = np.argmax(TestPredicted, axis=1)
        # Acc1, err1 = GetResults(predicted_labels_train, y_train)
        # Acc2, err2 = GetResults(predicted_labels_test, y_test)
        # ErrorListTrain.append(err1)
        # ErrorListTest.append(err2)
        # break
    # print(TestPredicted)
    # predicted_labels_train = np.argmax(TrainPredicted, axis=1)
    predicted_labels_test = np.argmax(TestPredicted, axis=1)
    # print(predicted_labels_test.shape)
    # Acc1, err1 = GetResults(predicted_labels_train, y_train)
    Acc2, err2 = GetResults(predicted_labels_test, y_test)
    # print("[Bagging Train] Accuracy:", Acc1)
    print("[Bagging Test] Accuracy:", Acc2)


