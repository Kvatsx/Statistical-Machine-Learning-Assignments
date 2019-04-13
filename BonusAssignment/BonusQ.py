# Kaustav Vats (2016048)

# https://www.kaggle.com/gauss256/preprocess-images/code
# https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from func import *
from sklearn.cluster import KMeans
import cv2
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Read Data ----------------------------------------------------
X_Data, Y_Data = ReadData(color=1)
print("Data Size:", X_Data.shape, "\nLabel Size:", Y_Data.shape)

# scaler = StandardScaler()
# scaler.fit(X_Data)
# X_Data = scaler.transform(X_Data)

# scaler = MinMaxScaler(feature_range=(0, 255))
# scaler.fit(X_Data)
# X_Data = scaler.transform(X_Data)

# https://www.kaggle.com/jfeng1023/data-cleaning-challenge-scale-and-normalize-data
# Normalized_Data = stats.zscore(X_Data)
# print(X_Data)

# Normalized_Data = stats.boxcox(X_Data)
# fig, ax=plt.subplots(1,2)
# sns.distplot(X_Data[0], ax=ax[0])
# ax[0].set_title("Original Data")
# sns.distplot(Normalized_Data[0], ax=ax[1])
# ax[1].set_title("Normalized data")

# # count = DataPerClass(X_Data, Y_Data)

# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def PreProcessing(x_data):
    NX_Data = []  
    for i in range(x_data.shape[0]):
        img = np.reshape(x_data[i], (64, 64, 3))
        # img = cv2.GaussianBlur(img,(3,3), 0)
        # img = cv2.equalizeHist(img)
        # img = cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)
# ----------------------
        mean = np.mean(img)
        std = np.std(img)
        img = (img-mean)/std
        img = img.astype(dtype=np.uint8)

# -----------------
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        final = cv2.merge((cl, a, b))
        img = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
# -----------------
        NX_Data.append(img.flatten())
    NX_Data = np.asarray(NX_Data)
    return NX_Data

X_Data = PreProcessing(X_Data)

# --------------------------------------------------------------------------
# X_Train, NX_Test, Y_Train, NY_Test = train_test_split(X_Data, Y_Data, test_size=0.3, random_state=42)
# count = DataPerClass(X_Train, Y_Train)

# Save & Load Data --------------------------------------------------------

# np.save("./BonusData/X_Train.npy", X_Train)
# np.save("./BonusData/Y_Train.npy", Y_Train)
# np.save("./BonusData/X_Test.npy", X_Test)
# np.save("./BonusData/Y_Test.npy", Y_Test)

print("Data Saved!")
# X_Train = np.load("./BonusData/X_Train.npy")
# Y_Train = np.load("./BonusData/Y_Train.npy")
# X_Test = np.load("./BonusData/X_Test.npy")
# Y_Test = np.load("./BonusData/Y_Test.npy")

# print("Train Size:", X_Train.shape, "\nLabels:", Y_Train.shape)
# print("Test Size:", X_Test.shape, "\nLabels:", Y_Test.shape)

# Feature Extraction ---------------------------------------------

def SiftFeatures(x_train, y_train):
    # orb = cv2.ORB_create()
    sift = cv2.xfeatures2d.SIFT_create()
    features = []
    nx_train = []
    ny_train = []
    for i in range(x_train.shape[0]):
        image = np.reshape(x_train[i], (64, 64, 3))
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(image, None)
        # _, des = orb.detectAndCompute(image, None)
        if des is None:
            continue
        for d in des:
            features.append(d)
        nx_train.append(x_train[i])
        ny_train.append(y_train[i])

    features = np.asarray(features)
    nx_train = np.asarray(nx_train)
    ny_train = np.asarray(ny_train)
    print("Features.shape", features.shape)

    return nx_train, ny_train, features 


# Model --------------------------------------------------------
# SVM Accuracy - 0.044848484848484846

print("-------------Features Collected----------------------")

# NX_Train = np.load("./BonusData/NX_Train.npy")
# NY_Train = np.load("./BonusData/NY_Train.npy")
# Features = np.load("./BonusData/Features.npy")

# print("Features.shape", Features.shape)

def getHisto(x_data):
    hds = []
    for i in range(x_data.shape[0]):
        img = np.reshape(x_data[i], (64, 64, 3))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        his = cv2.calcHist([img],[0],None,[256],[0,256])
        # print(his.shape)
        # break
        # his = np.reshape(his, (256, 1))
        hds.append(his)
    new_hist = np.zeros((x_data.shape[0], 256), dtype=np.int)
    for i in range(len(hds)):
        for j in range(hds[i].shape[0]):
            new_hist[i, j] = hds[i][j]
    return new_hist

def getHOG(x_data):
    hds = []
    for i in range(x_data.shape[0]):
        img = np.reshape(x_data[i], (64, 64))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True, block_norm="L2-Hys")
        hds.append(fd)
        cv2.imwrite("./BonusData/Visualize_Train/img_"+str(i)+".png", hog_image)
    hds = np.asarray(hds)
    print("Hog.shape", hds.shape)
    return hds

# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5)
# # clf = svm.SVC(gamma='auto')
# clf = DecisionTreeClassifier(random_state=0)

# for train_index, test_index in kf.split(X_Train):
#     NX_Train, NY_Train = X_Train[train_index], Y_Train[train_index]
#     NX_Test, NY_Test = X_Train[test_index], Y_Train[test_index]

#     NX_Train = getHisto(NX_Train)
#     NX_Test = getHisto(NX_Test)
    
#     clf.fit(NX_Train, NY_Train)
#     Accuracy = clf.score(NX_Train, NY_Train)
#     print("[DTC]Accuracy[Train]:", Accuracy*100)
#     Accuracy = clf.score(NX_Test, NY_Test)
#     print("[DTC]Accuracy[Test]:", Accuracy*100)
#     # clf.fit(NX_Train, NY_Train)
#     # Accuracy = clf.score(NX_Test, NY_Test)
#     # print("Accuracy:", Accuracy)

# X_Train_1 = getHisto(X_Train)
# NX_Test_1 = getHisto(NX_Test)
# X_Train = getHOG(X_Train)
# NX_Test = getHOG(NX_Test)
X_Train = getHisto(X_Data)
Y_Train = Y_Data
NX_Test, names = ReadTestData(color=1)
NX_Test = PreProcessing(NX_Test)
NX_Test = getHisto(NX_Test)

# np.save("./BonusData/X_Train.npy", X_Train_1)
# np.save("./BonusData/Y_Train.npy", NX_Test_1)
# np.save("./BonusData/X_Test.npy", X_Train)
# np.save("./BonusData/Y_Test.npy", NX_Test)

# print("Data Saved!")
# X_Train_1 = np.load("./BonusData/X_Train.npy")
# NX_Test_1 = np.load("./BonusData/Y_Train.npy")
# X_Train = np.load("./BonusData/X_Test.npy")
# NX_Test = np.load("./BonusData/Y_Test.npy")

# print("Hog Fe")
# print(X_Train.shape, X_Train_1.shape)


# X_Train = np.hstack((X_Train, X_Train_1))
# NX_Test = np.hstack((NX_Test, NX_Test_1))
print(X_Train.shape)
print(NX_Test.shape)

# pca = PCA()
# pca.fit(X_Train)
# X_Train =  pca.transform(X_Train)
# NX_Test = pca.transform(NX_Test)

# np.save("./BonusData/X_Train.npy", X_Train)
# np.save("./BonusData/Y_Train.npy", Y_Train)
# np.save("./BonusData/X_Test.npy", NX_Test)
# np.save("./BonusData/Y_Test.npy", NY_Test)

# X_Train =  np.load("./BonusData/X_Train.npy")
# Y_Train = np.load("./BonusData/Y_Train.npy")
# NY_Train = np.load("./BonusData/X_Test.npy")
# NY_Test = np.load("./BonusData/Y_Test.npy")

# Accuracy = clf.score(X_Train, Y_Train)
# print("Accuracy:", Accuracy*100)
# Accuracy = clf.score(X_Test, Y_Test)
# print("Accuracy:", Accuracy*100)

# X_Test, names = ReadTestData()
# X_Test = PreProcessing(X_Test)
# X_Train = getHisto(X_Data)
# Y_Train = Y_Data
# X_Test = getHisto(X_Test)
print("----------------HOG Features-----------------------")

# print("---------Sift Features-----------------------")
# size = int(X_Train.shape[0] * 0.5)
# print(X_Train.shape)
# X_Train_1 = X_Train[:size]
# X_Train_2 = X_Train[size:]

# print(X_Train_1.shape)
# print(X_Train_2.shape)
# print("Done")

# Y_Train_1 = Y_Train[:size]
# Y_Train_2 = Y_Train[size:]
# print(Y_Train_1.shape)
# print(Y_Train_2.shape)

# size = int(X_Data.shape[0]*0.5)
# X_Data_1 = X_Data[:size]
# X_Data_2 = X_Data[size:]

# Y_Data_1 = Y_Data[:size]
# Y_Data_2 = Y_Data[size:]

# X_Train_1, Y_Train_1, Features1 = SiftFeatures(X_Train_1, Y_Train_1)
# X_Train_1, Y_Train_1, Features1 = SiftFeatures(X_Data_1, Y_Data_1)



print("------------Clutering Features-----------------")

# kmeans = KMeans(n_clusters=50)
# kmeans.fit(Features1)
# pickle.dump(kmeans, open("./BonusData/Kmean.sav", 'wb'))
# kmeans = pickle.load(open("./BonusData/Kmean.sav", 'rb'))

# kmeans = pickle.load(open("./BonusData/Kmean.sav", 'rb'))
def Bovw(kmeans, x_data, y_data):
    sift = cv2.xfeatures2d.SIFT_create()
    # orb = cv2.ORB_create()
    Features = []
    Y_data = []
    for i in range(x_data.shape[0]):
        image = np.reshape(x_data[i], (64, 64, 3))
        histogram = np.zeros(len(kmeans.cluster_centers_))
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # kp, des = orb.detectAndCompute(image, None)
        kp, des = sift.detectAndCompute(image, None)
        # if des is None:
        #     continue
        nkp = np.size(kp)
        if not des is None:
            for d in des:
                idx = kmeans.predict(np.reshape(d, (1, d.shape[0])))
                histogram[idx] += 1
        Features.append(histogram)
        Y_data.append(y_data[i])

    Features = np.asarray(Features)
    Y_data = np.asarray(Y_data)
    return Features, Y_data
                             
# print("Vocab Test: ", X_Train_2.shape, Y_Train_2.shape)
# X_Train, Y_Train = Bovw(kmeans, X_Train_2, Y_Train_2)
# NX_Test, NY_Test = Bovw(kmeans, NX_Test, NY_Test)

# X_Train, Y_Train = Bovw(kmeans, X_Data_2, Y_Data_2)

# X_Test, names = ReadTestData()
# X_Test = PreProcessing(X_Test)
# NX_Test, names = Bovw(kmeans, X_Test, names)

print("------------Training Classifier-----------------")

# print(X_Train.shape, Y_Train.shape)
# svc = svm.SVC(gamma='auto')
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# clf = GridSearchCV(svc, parameters, cv=5)

# clf.fit(X_Train, Y_Train)
# pickle.dump(clf, open("./BonusData/Svm.sav", 'wb'))
# clf = pickle.load(open("./BonusData/Svm.sav", 'rb'))

print("-------------Training Complete------------------")

# NX_Test, NY_Test, _ = ExtractFeatures(X_Test, Y_Test)
# Accuracy = clf.score(X_Train, Y_Train)
# print("Accuracy[Train]:", Accuracy*100)
# Accuracy = clf.score(NX_Test, NY_Test)
# print("Accuracy[Test]:", Accuracy*100)
# labels = clf.predict(NX_Test)
# WriteCsv("2016048_kaustav_submission1.csv", names, labels)


clf = RandomForestClassifier(n_estimators=1200, random_state=0)
clf.fit(X_Train, Y_Train)
Accuracy = clf.score(X_Train, Y_Train)
print("[RFC]Accuracy[Train]:", Accuracy*100)
# labels = clf.predict(X_Test)

labels = clf.predict(NX_Test)
# WriteCsv("2016048_kaustav_submission2.csv", names, labels)

# np.save("./BonusData/labels.npy", labels)
# labels = np.load("./BonusData/labels.npy")
# print(names)
WriteCsv("2016048_kaustav_submission.csv", names, labels)
# Accuracy = clf.score(NX_Test, NY_Test)
# print("[RFC]Accuracy[Test]:", Accuracy*100)


# Accuracy = clf.score(X_Train, Y_Train)
# print("[DTC]Accuracy[Train]:", Accuracy*100)
# Accuracy = clf.score(X_Test, Y_Test)
# print("[DTC]Accuracy[Test]:", Accuracy*100)

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=50)
# clf.fit(X_Train, Y_Train)

# PredictedLabels = clf.predict(NX_Test)
# Accuracy = clf.score(X_Train, Y_Train)
# print("[KNN]Accuracy[Train]:", Accuracy*100)
# Accuracy = clf.score(NX_Test, NY_Test)
# print("[KNN]Accuracy[Test]:", Accuracy*100)


# clf = MLPClassifier(max_iter=600000)
# clf.fit(X_Train, Y_Train)
# Accuracy = clf.score(X_Train, Y_Train)
# print("[RFC]Accuracy[Train]:", Accuracy*100)
# labels = clf.predict(NX_Test)
# WriteCsv("2016048_kaustav_submission.csv", names, labels)

