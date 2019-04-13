# Kaustav Vats (2016048)

import numpy as np
from glob import glob
import csv
import cv2
import matplotlib.pyplot as plt

# def ReadData(color=1):
#     imageList = []
#     imageLabels = []
#     labels = {}
#     with open('./bonus_dataset/sml_train.csv', 'r') as file:
#         reader = csv.reader(file)
#         for line in reader:
#             labels[line[0]] = line[1]

#     folderImages = glob("./bonus_dataset/sml_train/*")
#     for item in folderImages:
#         img = cv2.imread(item)
#         if color == 0:
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         # mean = np.mean(img)
#         # std = np.std(img)
#         # img = (img-mean)/std
#         img2 = img.flatten()
#         imageList.append(img2)
#         item = item.split("\\")
#         imageLabels.append(labels[item[1]])

#     imageList = np.array(imageList)    
#     imageLabels = np.asarray(imageLabels, dtype=np.int64)
#     return imageList, imageLabels

def ReadData(color=1):
    imageList = []
    imageLabels = []
    with open('./bonus_dataset/sml_train.csv', 'r') as file:
        reader = csv.reader(file)
        flag = 1
        count = 0
        for line in reader:
            if flag:
                flag = 0
                continue
            imageLabels.append(line[1])
            img = cv2.imread("./bonus_dataset/sml_train/" + line[0], color)
            if count == 1:
                cv2.imwrite("./BonusData/final.png", img)
            imageList.append(img.flatten())
            count += 1
    
    # print(imageLabels)
    ReadTestData(color)
    imageList = np.asarray(imageList)
    imageLabels = np.asarray(imageLabels)
    return imageList, imageLabels


def ReadTestData(color=1):
    folderImages = glob("./bonus_dataset/sml_test/*")
    # print(folderImages[0])
    folderImages.sort(key=lambda x: int(x.split("_")[3].split(".")[0]))
    # print(folderImages)
    data = []
    names = []
    for item in folderImages:
        img = cv2.imread(item, color)
        names.append(item.split("\\")[1])
        data.append(img.flatten())
    data = np.asarray(data)
    names = np.asarray(names)
    return data, names



def DataPerClass(data, label):
    count = np.zeros(20, dtype=np.int64)
    for i in range(label.shape[0]):
        count[label[i]] += 1
    index = np.arange(count.shape[0])
    print("Data per class:\n", count)
    plt.bar(index, count)
    plt.xlabel('Classes', fontsize=5)
    plt.ylabel('Frequency', fontsize=5)
    plt.xticks(index, index, fontsize=5, rotation=30)
    plt.title('Class Data points Histogram')
    plt.show()
    return count


def WriteCsv(filename, names, label):
    results = []
    results.append(["Id", "Category"])
    for i in range(names.shape[0]):
        results.append([names[i], label[i]])
    results = np.asarray(results)
    print(results)

    with open('./BonusData/' + filename, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(results)
    csvFile.close()