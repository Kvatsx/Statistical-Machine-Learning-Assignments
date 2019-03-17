# Kaustav Vats (2016048)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from func import *

# Read Data ----------------------------------------------------
X_Data, Y_Data = ReadData(color=0)
print("Data Size:", X_Data.shape, "\nLabel Size:", Y_Data.shape)
print("No of classes:", np.unique(Y_Data).shape[0])

# count = DataPerClass(X_Data, Y_Data)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data, Y_Data, test_size=0.33, random_state=42)
# count = DataPerClass(X_Train, Y_Train)

# Save & Load Data --------------------------------------------------------

# np.save("./BonusData/X_Train.npy", X_Train)
# np.save("./BonusData/Y_Train.npy", Y_Train)
# np.save("./BonusData/X_Test.npy", X_Test)
# np.save("./BonusData/Y_Test.npy", Y_Test)

# X_Train = np.load("./BonusData/X_Train.npy")
# Y_Train = np.load("./BonusData/Y_Train.npy")
# X_Test = np.load("./BonusData/X_Test.npy")
# Y_Test = np.load("./BonusData/Y_Test.npy")

# print("Train Size:", X_Train.shape, "\nLabels:", Y_Train.shape)
# print("Test Size:", X_Test.shape, "\nLabels:", Y_Test.shape)

# Feature Extraction ---------------------------------------------

sift = cv2.xfeatures2d.SIFT_create()
NX_Train = []
NY_Train = []
NX_Test = []
NY_Test = []
for i in range(X_Train.shape[0]):
    kp, des = sift.detectAndCompute(X_Train[i],None)
    if des is None:
        continue
    # print(i, des.shape[0])
    des = np.reshape(des, (1, des.shape[0]*des.shape[1]))
    NX_Train.append(des)
    print(des.shape)
    NY_Train.append(Y_Train[i])

for i in range(X_Test.shape[0]):
    kp, des = sift.detectAndCompute(X_Test[i],None)
    if des is None:
        continue
    NX_Test.append(np.reshape(des, (1, des.shape[0]*des.shape[1])))
    NY_Test.append(Y_Test[i])

NX_Train = np.asarray(NX_Train)
NY_Train = np.asarray(NY_Train)
NX_Test = np.asarray(NX_Test)
NY_Test = np.asarray(NY_Test)

print("Samples:", NX_Train.shape)
# print(des.shape)


# Model --------------------------------------------------------
# SVM Accuracy - 0.044848484848484846

clf = svm.SVC(gamma='auto')
clf.fit(NX_Train, NY_Train)
Accuracy = clf.score(NX_Test, NY_Test)
print("Accuracy:", Accuracy)
