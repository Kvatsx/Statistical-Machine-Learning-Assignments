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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

X_Data, Y_Data = ReadData(color=1)

def PreProcessing(x_data):
    NX_Data = []  
    for i in range(x_data.shape[0]):
        img = np.reshape(x_data[i], (64, 64, 3))
        img = cv2.GaussianBlur(img,(3,3), 0)
        NX_Data.append(img.flatten())
    NX_Data = np.asarray(NX_Data)
    return NX_Data

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data, Y_Data, test_size=0.3, random_state=42)

print("[+] Features Extraction...")
# Sift Features
def SiftFeatures(x_data, y_data):
    sift = cv2.xfeatures2d.SIFT_create()
    features = []
    labels = []

    for i in range(x_data.shape[0]):
        image = np.reshape(x_data[i], (64, 64, 3))
        _, des = sift.detectAndCompute(image, None)
        if des is None:
            continue
        for d in des:
            features.append(d)
        labels.append(y_data[i])

    features = np.asarray(features)
    labels = np.asarray(labels)

    print("Features shape: {}".format(features.shape))
    print("Labels shape: {}".format(labels.shape))
    return features, labels

features, y_train = SiftFeatures(X_Train, Y_Train)

print("[+] Unsupervised Learning[KMeans]...")

kmeans = KMeans(n_clusters=100, n_jobs=-1)
kmeans.fit(features)
pickle.dump(kmeans, open("./BonusData/Kmean.sav", 'wb'))

# kmeans = pickle.load(open("./BonusData/Kmean.sav", 'rb'))
def Bovw(kmeans, x_data, y_data):
    sift = cv2.xfeatures2d.SIFT_create()
    # orb = cv2.ORB_create()
    Features = []
    Y_data = []
    for i in range(x_data.shape[0]):
        image = np.reshape(x_data[i], (64, 64, 3))
        histogram = np.zeros(len(kmeans.cluster_centers_))
        kp, des = sift.detectAndCompute(image, None)
        nkp = np.size(kp)

        if not des is None:
            for d in des:
                idx = kmeans.predict(np.reshape(d, (1, d.shape[0])))
                histogram[idx] += 1/nkp
        Features.append(histogram)
        Y_data.append(y_data[i])

    Features = np.asarray(Features)
    Y_data = np.asarray(Y_data)
    return Features, Y_data

X_Train, Y_Train = Bovw(kmeans, X_Train, Y_Train)
NX_Test, NY_Test = Bovw(kmeans, X_Test, Y_Test)

print("[+] Unsupervised Learning[KMeans] Done")


print("[+] Random Forest Classifier...")

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
clf.fit(X_Train, Y_Train)

# Accuracy = clf.score(NX_Train, NY_Train)
# print("[RFC]Accuracy[Train]:", Accuracy*100)

Accuracy = clf.score(NX_Test, NY_Test)
print("[RFC]Accuracy[Test]:", Accuracy*100)

print("[+] Random Forest Classifier done")
pickle.dump(clf, open("./BonusData/Clf.sav", 'wb'))

