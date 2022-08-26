import numpy as np

# n=3
# d=2
# K=2

X = np.array([[1, 2], [3, 4], [5, 6]])
print(f"{X[1]}\n\n\t\tvs\n\n{X[1, :]}")
# mu = np.array(np.arange(1, 5)).reshape(K, d)

# # print(X, '\n', mu)

# post = np.zeros((n, K))

# for j in range(K):
#     # post[:][j] =
#     print( X - mu[j] )        
#     pass

# # x, mu = np.array([[1, 2, 3, 4], [6, 7, 8, 9], [1, 4, 6, 8]]), np.array([10, 10, 10, 10])
# # num = X - mu
# # (n, d) - (K, d)
# # (n, K)
# #  
# den = np.sum(num, 0)
# # print(den)