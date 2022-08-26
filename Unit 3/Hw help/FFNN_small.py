import numpy as np

x = np.array([3, 14, 1]).reshape(3, 1)
w = np.array([1, 0, -1, 0, 1, -1, -1, 0, -1, 0, -1, -1]).reshape(4, 3)
v = np.array([1, 1, 1,1,0,-1,-1,-1,-1,2]).reshape(2, 5)

def relu(x):
    return np.maximum(0, x)

f = relu(w @ x)
# add a one to the end of f
f = np.append(f, 1)

f = relu(v @ f)
print(f)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

print(softmax(f))