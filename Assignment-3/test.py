import numpy as np

arr = np.array([[1, 2], [4, 6]])
mean = np.mean(arr, axis=0)
print (mean)

arr = arr/2
print (arr)

arr = np.array([2, 5, 3, 4])
print(np.argsort(-arr))

A = np.asarray([3, 9, 2, 24, 1, 6])
B = np.asarray(['a', 'b', 'c', 'd', 'e'])
print(sorted(zip(A, B)))
