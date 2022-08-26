import numpy as np


# a = np.arange(1, 16).reshape((3, 5))
b = np.arange(1, 6).reshape((5, 1))
print(b, b.T)
# print(a, b, sep='\n\n', end='\n\n\n')
# # print(np.dot(a, b))
# # print(a @ b)

# """compare if a and b are the same numpy array"""
# print(np.array_equal(a, b))

# print(a ^ 2)

"""

k = L = 1

U0 = np.array([6, 0, 3, 6])
V0 = np.array([4, 2, 1])

# u @ v.t
X0 = np.outer(U0, V0.T)

print('X0 is:', X0)

# np.nansum is the sum of all non-nan values
Y = np.array([[5, np.nan, 7], [np.nan, 2, np.nan], [4, np.nan, np.nan], [np.nan, 3, 6]])
# np.square is the square of each element

def squared_error_term(Y, X):
    return 0.5 * np.nansum(np.square(Y - X0))

def regularization_term(U, V, L):
    return 0.5 * L * np.nansum(np.square(U)) + 0.5 * L * np.nansum(np.square(V))  

print(squared_error_term(Y, X0))

# np.norm(matrix) is the Frobenius norm of the matrix (frobenius norm is sqrt of what we're using)
print(regularization_term(U0, V0, L))

# np.linspace(-2*np.pi, 2*np.pi, 400) is a 400-element vector with values between -2*pi and 2*pi


"""