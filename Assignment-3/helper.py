# Kaustav Vats (2016048)

import numpy as np
from glob import glob
import cv2
from numpy import linalg as la
from sklearn.naive_bayes import GaussianNB
from math import ceil
from random import randint as randi

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

def SplitData(X_Data, Y_Data):
    DataSize = X_Data.shape[0]
    Testing_Size = ceil(DataSize * 70/100)
    DataSlot = []
    print (Testing_Size, DataSize)
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
    mean = np.mean(data, axis=0)
    # print(mean)
    new_data = data - mean
    return new_data


def EigenValueDecomposition(data):
    CovarienceMatrix = np.cov(data.T)
    CovarienceMatrix = CovarienceMatrix/(data.shape[0]-1)
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

    M = np.matmul(la.inv(Sw), Sb)
    CovarienceMatrix = np.cov(M.T)
    EigenValues, EigenVector = la.eig(CovarienceMatrix)
    eigen_pair = []
    for i in range(len(EigenValues)):
        eigen_pair.append((np.abs(EigenValues[i]), EigenVector[:, i]))
    eigen_pair.sort(key=lambda k: k[0], reverse=True)
    
    W = []
    for i in range(len(eigen_pair)):
        W.append(np.real(eigen_pair[i][1]))

    W = np.asarray(W)
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
    finalMatrix = 0
    for i in range(classRange[0], classRange[1]):
        NewData = GetClassData(x_data, y_data, i)
        MeanData = np.mean(NewData, axis=0)
        result = MeanData - mean_value
        result = np.reshape(result, (result.shape[0], 1))
        # print(result)
        # print("result:", result.shape)
        result = np.matmul(result, result.T)
        # print(result)
        # print("result:", result.shape)
        result = result * NewData.shape[0]
        finalMatrix += result
    print("Sb:", finalMatrix.shape)
    # print(finalMatrix)
    return finalMatrix

def WithinClassScatter(x_data, y_data, classRange):
    finalMatrix = 0
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


# def FiveFoldCrossValidation(x_train, y_train, x_test, y_test):

