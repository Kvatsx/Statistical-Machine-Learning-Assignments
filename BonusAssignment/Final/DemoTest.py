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


def ReadData(color=1):
    imageList = []
    imageLabels = []
    with open('./demo/sml_val.csv', 'r') as file:
        reader = csv.reader(file)
        count = 0
        for line in reader:
            # print(line)
            imageLabels.append(line[1])
            img = cv2.imread("./demo/sml_validation/" + line[0], color)
            imageList.append(img.flatten())
            count += 1
    
    # print(imageLabels)
    ReadTestData(color)
    imageList = np.asarray(imageList)
    imageLabels = np.asarray(imageLabels)
    return imageList, imageLabels

def getfeatures(x_data):
    hds = []
    for i in range(x_data.shape[0]):
        img = np.reshape(x_data[i], (64, 64, 3))
        # his_b = cv2.calcHist([img],[0],None,[256],[0,256])
        # his_g = cv2.calcHist([img],[1],None,[256],[0,256])
        # his_r = cv2.calcHist([img],[1],None,[256],[0,256])

        # his_b = np.reshape(his_b, (1, 256))
        # his_g = np.reshape(his_g, (1, 256))
        # his_r = np.reshape(his_r, (1, 256))

        # final = np.hstack((his_b, his_g))
        # final = np.hstack((final, his_r))


        # his = np.reshape(his, (1, 256))
        # print("His.shape", his.shape)
        # 8x8 , 2x2
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, block_norm="L2-Hys", multichannel=True)
        # print(fd.shape)
        # fd = np.reshape(fd, (fd.shape[0], ))
        # hog_img = hog_image.flatten()        
        # hog_img = np.reshape(hog_img, (hog_img.shape[0], ))
        # his = np.hstack((his, fd))
        # his = np.hstack((fd, hog_img))
        # ---------------------------
        # mean = np.mean(img)
        # img = img - mean
        # contrast = np.sqrt(10 + np.mean(img**2))
        # img = img / max(contrast, 0.000000001)
        # # img = cv2.calcHist([img],[0],None,[256],[0,256])        
        # # img = cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)
        # img = img.flatten()        
        # img = np.reshape(img, (1, img.shape[0]))
        # his = np.hstack((his, img))
# -----------------
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img = hsv_img.flatten()
        # hsv_img = np.reshape(hsv_img, (hsv_img.shape[0], ))
        his = np.hstack((fd, hsv_img))

        hsv_h = cv2.calcHist([img],[0],None,[256],[0,256])
        hsv_s = cv2.calcHist([img],[1],None,[256],[0,256])
        hsv_v = cv2.calcHist([img],[2],None,[256],[0,256])

        hsv_h = np.reshape(hsv_h, (256, ))
        hsv_s = np.reshape(hsv_s, (256, ))
        hsv_v = np.reshape(hsv_v, (256, ))

        final2 = np.hstack((hsv_h, hsv_s))
        final2 = np.hstack((final2, hsv_v))
        # final2 = np.hstack((final2, fd))
        
        his = np.hstack((his, final2))
        # his = np.reshape(his, (his.shape[1], ))

        hds.append(his)
    
    hds = np.asarray(hds)
    print("Features shape: {}".format(hds.shape))
    return hds


test, label = ReadData()
print(test.shape)
print(label.shape)
Features = getfeatures(test)

with open("Clf.sav","rb") as f:
    clf = pickle.load(f)
    Acc = clf.score(Features, label)
    print("Acc:", Acc)

    


