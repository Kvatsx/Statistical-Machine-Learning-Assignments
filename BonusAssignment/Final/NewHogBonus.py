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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

X_Data, Y_Data = ReadData(color=1)

def PreProcessing(x_data):
    NX_Data = []  
    for i in range(x_data.shape[0]):
        img = np.reshape(x_data[i], (64, 64, 3))
        # img = cv2.GaussianBlur(img,(3,3), 0)
# -----------------
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
        # mean = np.mean(img)
        # img = img - mean
        # contrast = np.sqrt(10 + np.mean(img**2))
        # img = img / max(contrast, 0.000000001)
        # img = img.astype(dtype=np.uint8)
# -----------------
        # cv2.imshow("prev", img)
        # cv2.waitKey(0)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # img = cv2.equalizeHist(img)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # cv2.imshow("after", img)
        # cv2.waitKey(0)
# -----------------
        NX_Data.append(img.flatten())
    NX_Data = np.asarray(NX_Data)
    return NX_Data

X_Data = PreProcessing(X_Data)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data, Y_Data, test_size=0.3, random_state=42)


# https://github.com/arijitx/HandGesturePy/blob/master/svm_train.py
def HogFeatures(x_data):
    hogs = []
    winSize = (32, 32)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 12
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    # winStride = (8,8)
    # padding = (8,8)
    for i in range(x_data.shape[0]):
        img = np.reshape(x_data[i], (64, 64, 3))
        # hist = hog.compute(img, winStride, padding)
        hist = hog.compute(img)
        hist = np.reshape(hist, (hist.shape[0], ))
        hogs.append(hist)

    hogs = np.asarray(hogs)
    print("Features:", hogs.shape)
    return hogs

def getHoggy(x_data):
    hds = []
    for i in range(x_data.shape[0]):
        img = np.reshape(x_data[i], (64, 64, 3))
        fd, hog_image = hog(img, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, block_norm="L2-Hys")
        hds.append(fd)
    hds = np.asarray(hds)
    print("Hog.shape", hds.shape)
    return hds

def getHsvHistograms(x_data):
    hist = []
    for i in range(x_data.shape[0]):
        img = np.reshape(x_data[i], (64, 64, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv_h = cv2.calcHist([img],[0],None,[256],[0,256])
        hsv_s = cv2.calcHist([img],[1],None,[256],[0,256])
        hsv_v = cv2.calcHist([img],[1],None,[256],[0,256])

        hsv_h = np.reshape(hsv_h, (1, 256))
        hsv_s = np.reshape(hsv_s, (1, 256))
        hsv_v = np.reshape(hsv_v, (1, 256))

        final = np.hstack((hsv_h, hsv_s))
        final = np.hstack((final, hsv_v))
        
        final = np.reshape(final, (final.shape[1], ))
        hist.append(final)
    hist = np.asarray(hist)
    return hist

def getFlatten(x_data):
    imgs = []
    for i in range(x_data.shape[0]):
        img = x_data[i].flatten()
        imgs.append(img)
    imgs = np.asarray(imgs)
    return imgs

NX_Train = getHoggy(X_Train)
NX_Test = getHoggy(X_Test)

pca = PCA(n_components=100)
pca.fit(NX_Train)
NX_Train =  pca.transform(NX_Train)
NX_Test = pca.transform(NX_Test)

NY_Test = Y_Test

nx_train_2 = getFlatten(X_Train)
nx_test_2 = getFlatten(X_Test)

pca = PCA(n_components=100)
pca.fit(nx_train_2)
nx_train_2 =  pca.transform(nx_train_2)
nx_test_2 = pca.transform(nx_test_2)

NX_Train = np.hstack((NX_Train, nx_train_2))
NX_Test = np.hstack((NX_Test, nx_test_2))

# hsv = getHsvHistograms(X_Train)
# hsv_test = getHsvHistograms(X_Test)

# X_Train = np.hstack((NX_Train, hsv))
# NX_Test = np.hstack((NX_Test, hsv_test))

# pca = PCA(0.99)
# pca.fit(X_Train)
# X_Train =  pca.transform(X_Train)
# NX_Test = pca.transform(NX_Test)

print("[PCA]Features:", NX_Train.shape)


print("[+] Random Forest Classifier...")

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
# Accuracy = clf.score(NX_Train, NY_Train)
# print("[RFC]Accuracy[Train]:", Accuracy*100)




# clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
# clf1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
# clf2 = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
# clf3 = MLPClassifier(max_iter=600000)

# clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('mlp', clf3)], voting='soft', n_jobs=-1)
clf.fit(NX_Train, Y_Train)
Accuracy = clf.score(NX_Test, NY_Test)

print("[RFC]Accuracy[Test]:", Accuracy*100)

print("[+] Random Forest Classifier done")
