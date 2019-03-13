# Kaustav Vats (2016048)

# In[]
import numpy as np
from numpy import linalg as la
from helper import ReadData1, SplitData, GaussianClassifier, NormalizeData, EigenValueDecomposition, EigenValueProjection, ProjectedData, PCA, LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# In[]
# X_Data, Y_Data = ReadData1()

# print("Data:", X_Data.shape, Y_Data.shape)
# # print(X_Data)
# X_Train, Y_Train, X_Test, Y_Test = SplitData(X_Data, Y_Data)
# GaussianClassifier(X_Train, Y_Train, X_Test, Y_Test)

# X_Train, Y_Train, X_Test, Y_Test = SplitData(X2_Data, Y_Data)
# print(X_Train.shape, X_Test.shape)
# np.save("Q1_Data\X_Train.npy", X_Train)
# np.save("Q1_Data\Y_Train.npy", Y_Train)
# np.save("Q1_Data\X_Test.npy", X_Test)
# np.save("Q1_Data\Y_Test.npy", Y_Test)

# In[]
# Load Data
X_Train = np.load("Q1_Data\X_Train.npy")
Y_Train = np.load("Q1_Data\Y_Train.npy")
X_Test = np.load("Q1_Data\X_Test.npy")
Y_Test = np.load("Q1_Data\Y_Test.npy")

print("Original Data:", X_Train.shape, X_Test.shape)

GaussianClassifier(X_Train, Y_Train, X_Test, Y_Test)

# PCA 
X2_Train, X2_Test = PCA(X_Train, X_Test, 95)
print("[PCA] Projected Data", X2_Train.shape, X2_Test.shape)
GaussianClassifier(X2_Train, Y_Train, X2_Test, Y_Test)

# LDA
X3_Train, X3_Test = LDA(X_Train, Y_Train, X_Test)
print("[LDA] Projected Data", X3_Train.shape, X3_Test.shape)
GaussianClassifier(X3_Train, Y_Train, X3_Test, Y_Test)

# clf = LinearDiscriminantAnalysis()
# clf.fit(X_Train, Y_Train)
# Predicted = clf.predict(X_Test)
# count2 = 0
# # print(Predicted.shape)
# for i in range(len(Y_Test)):
#     if (Y_Test[i] == Predicted[i]):
#         count2 += 1
# print("Accuracy[Test]:", (count2/len(Y_Test))*100) 




