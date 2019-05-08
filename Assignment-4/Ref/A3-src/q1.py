
# coding: utf-8

# In[73]:


import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[51]:


mnistdata = h5.File('./HW3_NN/data/Q1/MNIST_Subset.h5','r')
dataX = np.array(mnistdata['X'])
dataY = np.array(mnistdata['Y'])


# ## Preprocessing

# In[52]:


dataX = np.reshape(dataX,(dataX.shape[0],dataX.shape[1]*dataX.shape[2]))
#----------Train and test split-----------------#
ordering = np.arange(len(dataX))
np.random.shuffle(ordering)
print(ordering)
randX = dataX[ordering]
randY = dataY[ordering]
trainX,trainY,testX,testY = randX[:10000],randY[:10000],randX[10001:],randY[10001:]
scaler = MinMaxScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.fit_transform(testX)
#one hot encoding
trainY = (trainY==7).astype(int)
labels = (trainY==0).astype(int)
trainY = np.vstack((np.array([trainY]),np.array([labels])))
trainY=trainY.T
print(trainY.shape)
testY = (testY==7).astype(int)
labels = (testY==0).astype(int)
testY = np.vstack((np.array([testY]),np.array([labels])))
testY=testY.T
print(testY.shape)


# In[53]:


def sigmoid(z):
    return 1/(1+np.exp(-z))
def softmax(z):
    exponent = np.exp(z)
    sums = np.sum(exponent,axis=1,keepdims=True)
    return exponent / sums


# In[54]:


def network(layers,units,trainX,trainY,learning_rate):
#     W1 = np.random.uniform(-1,1,(784,100))
#     b1 = np.random.uniform(-1,1,(100))
#     W2 = np.random.uniform(-1,1,(100,50))
#     b2 = np.random.uniform(-1,1,(50))
#     W3 = np.random.uniform(-1,1,(50,50))
#     b3 = np.random.uniform(-1,1,(50))
#     W4 = np.random.uniform(-1,1,(50,2))
#     b4 = np.random.uniform(-1,1,(2))
    features = [784] + units + [2]
    W = []
    b = []
    print("Initializing weights and biases")
    for i in range(len(features)-1):
        w = np.random.uniform(-1,1,(features[i],features[i+1]))
        bias = np.random.uniform(-1,1,(features[i+1]))
        print(w.shape)
        print(bias.shape)
        W.append(w)
        b.append(bias)
    train_error =[]
    for epochs in tqdm(range(1000)):
        #------Forward Propogation------------#
#         z1 = np.dot(trainX,W1) + b1
#         a1 = sigmoid(z1)
#         z2 = np.dot(a1,W2) + b2
#         softmaxout = softmax(z2)

#         z1 = np.dot(trainX,W1) + b1
#         a1 = sigmoid(z1)
#         z2 = np.dot(a1,W2) + b2
#         a2 = sigmoid(z2)
#         z3 = np.dot(a2,W3) + b3
#         a3 = sigmoid(z3)
#         z4 = np.dot(a3,W4) + b4
#         softmaxout = softmax(z4)

        a = []
        for i in range(layers):
            if(i==0):
                z = np.dot(trainX,W[i]) + b[i]
                activation = sigmoid(z)
                a.append(activation)
            else:
                z = np.dot(a[i-1],W[i]) + b[i]
                activation =  sigmoid(z)
                a.append(activation)
        z = np.dot(a[layers-1],W[layers]) + b[layers]
        softmaxout = softmax(z)
        
        
#         -------backprop-------#
        delta = softmaxout-trainY
        dW = (1/10000)*np.dot(a[layers-1].T,softmaxout-trainY)
        db = (1/10000)*np.sum(softmaxout-trainY)
        W[layers] = W[layers] - learning_rate*dW
        b[layers] = b[layers] - learning_rate*db
        deltas= []
        deltas.append(delta)
        i=layers-1
        while i>=0:
            delta = np.dot(delta,W[i+1].T) * a[i] * (1 - a[i])
            deltas.append(delta)
            i=i-1
        deltas.reverse()
        i=layers-1
        while i>=1:
            W[i] = W[i] - (1/10000)*learning_rate*np.dot(a[i-1].T,deltas[i])
            b[i] = b[i] - (1/10000)*learning_rate*np.sum(deltas[i],axis=0)
            i=i-1
        W[i] = W[i] - (1/10000)*learning_rate*np.dot(trainX.T,deltas[i])
        b[i] = b[i] - (1/10000)*learning_rate*np.sum(deltas[i],axis=0)
        #1 LAYER
#         delta2 = softmaxout-trainY
#         dW = (1/10000)*np.dot(a1.T,delta2)
#         db = (1/10000)*np.sum(delta2,axis=0)
#         W2 = W2 - learning_rate*dW
#         b2 = b2 - learning_rate*db
#         delta1 = np.dot(delta2,W2.T) * a1 * (1 - a1)
#         dW = (1/10000)*np.dot(trainX.T,delta1)
#         db = (1/10000)*np.sum(delta1,axis=0)
#         W1 = W1 - learning_rate*dW
#         b1 = b1 - learning_rate*db
        
        #3 LAYERS 
#         delta4 = softmaxout-trainY
#         dW = (1/10000)*np.dot(a3.T,delta4)
#         db = (1/10000)*np.sum(delta4,axis=0)
#         W4 = W4 - learning_rate*dW
#         b4 = b4 - learning_rate*db
#         delta3 = np.dot(delta4,W4.T) * a3 * (1 - a3)
#         dW = (1/10000)*np.dot(a2.T,delta3)
#         db = (1/10000)*np.sum(delta3,axis=0)
#         W3 = W3 - learning_rate*dW
#         b3 = b3 - learning_rate*db
#         delta2 = np.dot(delta3,W3.T) * a2 * (1 - a2)
#         dW = (1/10000)*np.dot(a1.T,delta2)
#         db = (1/10000)*np.sum(delta2,axis=0)
#         W2 = W2 - learning_rate*dW
#         b2 = b2 - learning_rate*db
#         delta1 = np.dot(delta2,W2.T) * a1 * (1 - a1)
#         dW = (1/10000)*np.dot(trainX.T,delta1)
#         db = (1/10000)*np.sum(delta1,axis=0)
#         W1 = W1 - learning_rate*dW
#         b1 = b1 - learning_rate*db
        
        #----------------------training error calculation-----------#
        if(epochs%30==0):
#             error = np.square(softmaxout-trainY).T[0]
#             total_error = np.sqrt((1/10000)*np.sum(error))
            loss = np.sum(-trainY * np.log(softmaxout))
            train_error.append(loss)
            print(loss)
        
    epochs = np.arange(len(train_error))
    plt.plot(epochs,train_error)
#     plt.savefig('sigmoid_3layer.png')
    plt.show()
    return W,b


# In[55]:


def predict(testX,W,b):
    a=[]
    for i in range(len(W)-1):
        if(i==0):
            z = np.dot(testX,W[i]) + b[i]
            activation = sigmoid(z)
            a.append(activation)
        else:
            z = np.dot(a[i-1],W[i]) + b[i]
            activation = sigmoid(z)
            a.append(activation)
    z = np.dot(a[len(a)-1],W[len(b)-1]) + b[len(b)-1]
    softmaxout = softmax(z)
    return softmaxout


# In[56]:


def acc(predictions,labels):
    count=0
    for vec1,vec2 in zip(predictions,labels):
        binary = (vec1==max(vec1)).astype(int)
        if((binary==vec2).all()):
            count+=1
    print(count/len(labels))


# In[63]:


w,b = network(3,[100,50,50],trainX,trainY,.9)


# In[64]:


predictions = predict(testX,w,b)
acc(predictions,testY)
predictions = predict(trainX,w,b)
acc(predictions,trainY)


# In[7]:


def relu(z):
    z = z*(z>0)
    return z


# In[40]:


def predict_relu(testX,W,b):
    a=[]
    for i in range(len(W)-1):
        if(i==0):
            z = np.dot(testX,W[i]) + b[i]
            activation = relu(z)
            a.append(activation)
        else:
            z = np.dot(a[i-1],W[i]) + b[i]
            activation = relu(z)
            a.append(activation)
    z = np.dot(a[len(a)-1],W[len(b)-1]) + b[len(b)-1]
    softmaxout = softmax(z)
    return softmaxout


# In[76]:


def network_relu(layers,units,trainX,trainY,learning_rate):
    features = [784] + units + [2]
    W = []
    b = []
    print("Initializing weights and biases")
    for i in range(len(features)-1):
        w = np.random.uniform(-1,1,(features[i],features[i+1]))
        bias = np.random.uniform(-1,1,(features[i+1]))
        print(w.shape)
        print(bias.shape)
        W.append(w)
        b.append(bias)
    train_error =[]
    for epochs in tqdm(range(8000)):
        #------Forward Propogation------------#
        a = []
        for i in range(layers):
            if(i==0):
                z = np.dot(trainX,W[i]) + b[i]
                activation = relu(z)
                a.append(activation)
            else:
                z = np.dot(a[i-1],W[i]) + b[i]
                activation =  relu(z)
                a.append(activation)
        z = np.dot(a[layers-1],W[layers]) + b[layers]
        softmaxout = softmax(z)
#         -------backprop-------#
        delta = softmaxout-trainY
        dW = (1/10000)*np.dot(a[layers-1].T,softmaxout-trainY)
        db = (1/10000)*np.sum(softmaxout-trainY)
        W[layers] = W[layers] - learning_rate*dW
        b[layers] = b[layers] - learning_rate*db
        deltas= []
        deltas.append(delta)
        i=layers-1
        while i>=0:
            relu_derivative = (a[i]>0).astype(int)
            delta = np.dot(delta,W[i+1].T) * relu_derivative
            deltas.append(delta)
            i=i-1
        deltas.reverse()
        i=layers-1
        while i>=1:
            W[i] = W[i] - (1/10000)*learning_rate*np.dot(a[i-1].T,deltas[i])
            b[i] = b[i] - (1/10000)*learning_rate*np.sum(deltas[i],axis=0)
            i=i-1
        W[i] = W[i] - (1/10000)*learning_rate*np.dot(trainX.T,deltas[i])
        b[i] = b[i] - (1/10000)*learning_rate*np.sum(deltas[i],axis=0)
        
        #----------------------training error calculation-----------#
        if(epochs%30==0):
#             error = np.square(softmaxout-trainY).T[0]
#             total_error = np.sqrt((1/10000)*np.sum(error))
            loss = np.sum(-trainY * np.log(softmaxout))
            train_error.append(loss)
            acc(softmaxout,trainY)
    epochs = np.arange(len(train_error))
    plt.plot(epochs,train_error)
#     plt.savefig('relu_1layer.png')
    plt.show()
    return W,b


# In[77]:


w,b = network_relu(3,[100,50,50],trainX,trainY,.0001)


# In[78]:


predictions = predict_relu(testX,w,b)
acc(predictions,testY)
predictions = predict_relu(trainX,w,b)
acc(predictions,trainY)


# In[79]:


pickle_out = open('weights_relu3.pickle','wb')
pickle.dump(w,pickle_out)
pickle.dump(b,pickle_out)
pickle_out.close()


# In[81]:


pickle_in = open('weights_sigmoidl1.pickle','rb')
w = pickle.load(pickle_in)
b = pickle.load(pickle_in)
predictions = predict(testX,w,b)
acc(predictions,testY)
predictions = predict(trainX,w,b)
acc(predictions,trainY)
pickle_in.close()


pickle_in = open('weights_sigmoidl3.pickle','rb')
w = pickle.load(pickle_in)
b = pickle.load(pickle_in)
predictions = predict(testX,w,b)
acc(predictions,testY)
predictions = predict(trainX,w,b)
acc(predictions,trainY)
pickle_in.close()


pickle_in = open('weights_relu1.pickle','rb')
w = pickle.load(pickle_in)
b = pickle.load(pickle_in)
predictions = predict_relu(testX,w,b)
acc(predictions,testY)
predictions = predict_relu(trainX,w,b)
acc(predictions,trainY)
pickle_in.close()


pickle_in = open('weights_relu3.pickle','rb')
w = pickle.load(pickle_in)
b = pickle.load(pickle_in)
predictions = predict_relu(testX,w,b)
acc(predictions,testY)
predictions = predict_relu(trainX,w,b)
acc(predictions,trainY)
pickle_in.close()


# In[75]:


trainYnew = trainY.T[0]
testYnew = testY.T[0]
print(trainYnew)
print(testYnew)
C=[0.01,.1,1,10]
params = {'C':C}
grid = GridSearchCV(SVC(),params,cv=3)
grid.fit(trainX,trainYnew)
clf = SVC(C = grid.best_params_['C'])
clf.fit(trainX,trainYnew)
print(clf.score(testX,testYnew))


# In[82]:


print(grid.best_params_['C'])


# In[83]:


print(clf.score(trainX,trainYnew))


# In[84]:


pickle_out = open('mnist_svm_grid.pickle','wb')
pickle.dump(grid,pickle_out)
pickle_out.close()
pickle_out = open('mnist_svm_clf.pickle','wb')
pickle.dump(clf,pickle_out)
pickle_out.close()


# In[85]:


pickle_in = open('mnist_svm_grid.pickle','rb')
grid = pickle.load(pickle_in)
pickle_in = open('mnist_svm_clf.pickle','rb')
clf = pickle.load(pickle_in)

