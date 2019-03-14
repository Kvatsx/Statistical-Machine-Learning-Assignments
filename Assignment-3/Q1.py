# Kaustav Vats (2016048)

# In[]
import numpy as np
from numpy import linalg as la
from helper import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# In[] ----------------------------------------
X_Data, Y_Data = ReadData1()

# print("Data:", X_Data.shape, Y_Data.shape)
# # print(X_Data)
X_Train, Y_Train, X_Test, Y_Test = SplitData(X_Data, Y_Data, 70)
# GaussianClassifier(X_Train, Y_Train, X_Test, Y_Test)

# X_Train, Y_Train, X_Test, Y_Test = SplitData(X2_Data, Y_Data, 70)
# print(X_Train.shape, X_Test.shape)
# np.save("Q1_Data\X_Train.npy", X_Train)
# np.save("Q1_Data\Y_Train.npy", Y_Train)
# np.save("Q1_Data\X_Test.npy", X_Test)
# np.save("Q1_Data\Y_Test.npy", Y_Test)

# In[]
# Load Data ----------------------------------------------
# X_Train = np.load("Q1_Data\X_Train.npy")
# Y_Train = np.load("Q1_Data\Y_Train.npy")
# X_Test = np.load("Q1_Data\X_Test.npy")
# Y_Test = np.load("Q1_Data\Y_Test.npy")

print("Original Data:", X_Train.shape, X_Test.shape)

GaussianClassifier(X_Train, Y_Train, X_Test, Y_Test)

# # PCA ---------------------------------------------------
X2_Train, X2_Test = PCA(X_Train, X_Test, 95)
print("[PCA] Projected Data", X2_Train.shape, X2_Test.shape)
GaussianClassifier(X2_Train, Y_Train, X2_Test, Y_Test)

# # LDA ---------------------------------------------------
X3_Train, X3_Test = LDA(X_Train, Y_Train, X_Test)
print("[LDA] Projected Data", X3_Train.shape, X3_Test.shape)
GaussianClassifier(X3_Train, Y_Train, X3_Test, Y_Test)

# # Inbuilt LDA ---------------------------------------------------
clf = LinearDiscriminantAnalysis()
clf.fit(X_Train, Y_Train)
Predicted = clf.predict(X_Test)
count2 = 0
# print(Predicted.shape)
for i in range(len(Y_Test)):
    if (Y_Test[i] == Predicted[i]):
        count2 += 1
print("Accuracy[Test]:", (count2/len(Y_Test))*100) 

# N Fold CrossValidation ---------------------------------------------------

# X_Train_Bins, Y_Train_Bins = NFold(5, X2_Train, Y_Train, X2_Test, Y_Test)
# X_Train_Bins, Y_Train_Bins = NFold(5, X3_Train, Y_Train, X3_Test, Y_Test)

X_Train_Bins, Y_Train_Bins = NFold(5, X_Train, Y_Train, X_Test, Y_Test)
# print(len(X_Train_Bins[0]))
def NFoldCrossValidation(x_train_bin, y_train_bin):
    Accuracy = []
    for i in range(len(x_train_bin)):
        new_x_train = []
        new_y_train = []
        new_x_test = []
        new_y_test = []
        for j in range(len(x_train_bin)):
            if i != j:
                for k in range(len(x_train_bin[j])):
                    new_x_train.append(x_train_bin[j][k])
                    new_y_train.append(y_train_bin[j][k])
        
        new_x_train = np.asarray(new_x_train)
        new_y_train = np.asarray(new_y_train)
        new_x_test = x_train_bin[i]
        new_y_test = y_train_bin[i]

        acc = GaussianClassifier(new_x_train, new_y_train, new_x_test, new_y_test)
        Accuracy.append(acc)
        # print(new_x_train)
        print(new_x_train.shape)        
        print(new_x_test.shape)

    print("Best Accuracy:", Accuracy[np.argmax(Accuracy)])

# NFoldCrossValidation(X_Train_Bins, Y_Train_Bins)

# 

