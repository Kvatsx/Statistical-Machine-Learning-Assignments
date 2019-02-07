# @Author: Kaustav Vats 
# @Roll-Number: 2016048 

#%%
import numpy as np
import matplotlib.pyplot as plt
from math import exp, pi, sqrt, log

#%%
def BB(pw1, pw2, u1, u2, s1, s2):
    expo = (1/8)*(u2-u1)*((s1+s2)/2)*(u2-u1)+(1/2)*(log((s1+s2)/sqrt(abs(s1)*abs(s2))))
    prod = sqrt(pw1*pw2)
    return prod*expo

def RandomPoints(u, v, count):
    return np.random.normal(u, v, count)

def Likelihood(p, u, v):
    twov = 2*v
    num = exp((-1)*((p-u)**2)/twov)
    den = sqrt(pi*twov)
    return num/den

def Predict(test, u1, v1, u2, v2):
    current = 0
    Correct = 0
    for i in range(test.shape[0]):
        lk1 = Likelihood(test[i], u1, v1)
        lk2 = Likelihood(test[i], u2, v2)
        if i >= 10:
            current = 1
        if lk1 > lk2:
            if current == 0:
                Correct += 1
        else:
            if current == 1:
                Correct += 1
    ret = [(Correct/test.shape[0])*100, ((test.shape[0]-Correct)/test.shape[0])*100]
    return ret

#%%
given = [
    [0.5, 0.5, -0.5, 0.5, 1, 1],
    [2/3, 1/3, -0.5, 0.5, 2, 2],
    [0.5, 0.5, -0.5, 0.5, 2, 2],
    [0.5, 0.5, -0.5, 0.5, 3, 1]
]
PopCount = [10, 50, 100, 200, 500, 1000]
for i in range(len(given)):
    print("I:", i)
    for j in PopCount:
        pw1 = given[i][0]
        pw2 = given[i][1]
        u1 = given[i][2]
        u2 = given[i][3]
        v1 = given[i][4]
        v2 = given[i][5]

        perror = BB(pw1, pw2, u1, u2, v1, v2)
        print(perror)
        points1 = RandomPoints(u1, v1, j)
        points2 = RandomPoints(u2, v2, j)
        # print(points1, points2)
        points = np.concatenate((points1, points2))
        # print(points, points.shape)
        retVal = Predict(points, u1, v1, u2, v2)
        print("Acc:", retVal[0])
        print("Empirical Error:", retVal[1])

    # break