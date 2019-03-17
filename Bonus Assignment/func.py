# Kaustav Vats (2016048)

import numpy as np
from glob import glob
import csv
import cv2

def ReadData(color=0):
    imageList = []
    imageLabels = []
    folderImages = glob("./bonus_dataset/sml_train/*")
    for item in folderImages:
        img = cv2.imread(item)
        if color == 0:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # img = img.flatten()
        imageList.append(img)
    with open('./bonus_dataset/sml_train.csv', 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            imageLabels.append(line[1])
        imageLabels.pop(0)
    imageList = np.array(imageList)
    imageLabels = np.asarray(imageLabels, dtype=np.int64)
    return imageList, imageLabels

def DataPerClass(data, label):
    count = np.zeros(20, dtype=np.int64)
    for i in range(label.shape[0]):
        count[label[i]] += 1
    print("Data per class:\n", count)
    return count
