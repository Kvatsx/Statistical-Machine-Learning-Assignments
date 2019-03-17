# Kaustav Vats (2016048)

# In[]
import numpy as np
from numpy import linalg as la
from helper import *
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ----------------------------------------
# X_Data, Y_Data = ReadData_1Q1()
# X_Train, Y_Train, X_Test, Y_Test = ReadData_2Q1()
# print(X_Data.shape)
# print(Y_Data.shape)
# # print("Data:", X_Data.shape, Y_Data.shape)
# # # print(X_Data)
# X_Train, Y_Train, X_Test, Y_Test = SplitData(X_Data, Y_Data, 70)
# GaussianClassifier(X_Train, Y_Train, X_Test, Y_Test)

# # X_Train, Y_Train, X_Test, Y_Test = SplitData(X_Data, Y_Data, 70)
# print(X_Train.shape, X_Test.shape)
# np.save("Q1_Data\X_Train.npy", X_Train)
# np.save("Q1_Data\Y_Train.npy", Y_Train)
# np.save("Q1_Data\X_Test.npy", X_Test)
# np.save("Q1_Data\Y_Test.npy", Y_Test)

# Load Data ----------------------------------------------
X_Train = np.load("Q1_Data\X_Train.npy")
Y_Train = np.load("Q1_Data\Y_Train.npy")
X_Test = np.load("Q1_Data\X_Test.npy")
Y_Test = np.load("Q1_Data\Y_Test.npy")

# print("Original Data:", X_Train.shape, X_Test.shape)

# GaussianClassifier(X_Train, Y_Train, X_Test, Y_Test)
# print(np.unique(Y_Test))
# # PCA ---------------------------------------------------
X2_Train, X2_Test, W = PCA(X_Train, X_Test, 99)
# print("[PCA] Projected Data", X2_Train.shape, X2_Test.shape)
# GaussianClassifier(X2_Train, Y_Train, X2_Test, Y_Test)
# print(W[0].shape)
# FaceImages = ReconstructImages(W)
# VisualizeEigenFaces(FaceImages)

# # LDA ---------------------------------------------------
X3_Train, X3_Test = LDA(X_Train, Y_Train, X_Test, [0, 10])    # Q2
# X3_Train, X3_Test = LDA(X_Train, Y_Train, X_Test, [0, 11]) # Q1
# X2_Train, X2_Test, W = PCA(X3_Train, X3_Test, 99)

# print("[LDA] Projected Data", X3_Train.shape, X3_Test.shape)
# print("[LDA] Projected Data", X3_Train.shape, X3_Test.shape)
# GaussianClassifier(X3_Train, Y_Train, X3_Test, Y_Test)
GaussianClassifier(X_Train, Y_Train, X_Test, Y_Test)


# # Inbuilt LDA ---------------------------------------------------
# clf = LinearDiscriminantAnalysis()
# clf.fit(X_Train, Y_Train)
# Predicted = clf.predict(X_Test)
# count2 = 0
# # print(Predicted.shape)
# for i in range(len(Y_Test)):
#     if (Y_Test[i] == Predicted[i]):
#         count2 += 1
# print("Accuracy[Test]:", (count2/len(Y_Test))*100) 

# N Fold CrossValidation ---------------------------------------------------

# X_Train_Bins, Y_Train_Bins = NFold(5, X2_Train, Y_Train, X2_Test, Y_Test)
# X_Train_Bins, Y_Train_Bins = NFold(5, X3_Train, Y_Train, X3_Test, Y_Test)

# X_Train_Bins, Y_Train_Bins = NFold(5, X3_Train, Y_Train, X3_Test, Y_Test)
# np.save("Q1_Data\X_Train_Bins_pca.npy", X_Train_Bins)
# np.save("Q1_Data\Y_Train_Bins_pca.npy", Y_Train_Bins)
# print(len(X_Train_Bins[0]))

# X_Train_Bins = np.load("Q1_Data\X_Train_Bins.npy")
# Y_Train_Bins = np.load("Q1_Data\Y_Train_Bins.npy")

# NFoldCrossValidation(X_Train_Bins, Y_Train_Bins, X2_Test, Y_Test)

# 

