import numpy as np

A = np.array([[[1, -3, 3], [3, -5, 3], [6, -6, 4]], [[1, -3, 3], [3, -5, 3], [6, -6, 4]], [[1, -3, 3], [3, -5, 3], [6, -6, 4]]])
B = np.array([[4, -2, -2], [4, -2, -2], [4, -2, -2]])
E = np.linalg.eigvals(A)
#print A
print B
print np.linalg.eigvals(B)
#print E
#print np.linalg.eigvals(E)
#print np.linalg.eigvals(D)
