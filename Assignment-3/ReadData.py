
def ReadData_1Q1():
    imageList = []
    imageLabels = []
    for i in range(1, 12):
        folderImages = glob(".\\Q1_dataset\\Face_data\\"+str(i)+"\\*")
        for item in folderImages:
            img = cv2.imread(item, 0)
            img = cv2.resize(img, (50, 50))
            imageList.append(img.flatten())
            imageLabels.append(i-1)
        # print(np.array(imageList).shape)

    imageList = np.array(imageList)
    # print(imageList.shape)
    imageLabels = np.asarray(imageLabels)
    return imageList, imageLabels

def ReadData_2Q1():
    data = []
    label = []
    for i in range(1, 6):
        with open(".\\Q1_dataset\\cifar-10-batches-py\\data_batch_" + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            data.append(dict[b'data'])
            label.append(dict[b'labels'])

    for i in range(1, len(data)):
        np.concatenate((data[0], data[i]), axis = 0)
        np.concatenate((label[0], label[i]), axis = 0)

    test = []
    test_label = []
    with open(".\\Q1_dataset\\cifar-10-batches-py\\test_batch", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        test.append(dict[b'data'])
        test_label.append(dict[b'labels'])

    data = np.asarray(data[0])
    label = np.asarray(label[0])
    test = np.asarray(test[0])
    test_label = np.asarray(test_label[0])
    print(data.shape)
    print(test.shape)
    # data = RGB2Gray(data, 32)
    # test = RGB2Gray(test, 32)
    return data, label, test, test_label 

def RGB2Gray(data, dim):
    new_data = np.zeros((data.shape[0], dim*dim), dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(dim*dim):
            new_data[i, j] = data[i, j] * 0.299 + data[i, j + dim*dim] * 0.587 + data[i, j + dim*dim*2] * 0.114
    print(new_data.shape)
    return new_data
