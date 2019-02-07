import numpy as np

mat1 = np.array([1 ,2])
mat2 = np.array([[1, 2], [3, 4]])
mat3 = mat1
mat4 = np.array([1, 2], )
print(mat4)
print(np.transpose(mat1))
print(np.matmul(np.transpose(mat1), mat2))
print(mat1/2)