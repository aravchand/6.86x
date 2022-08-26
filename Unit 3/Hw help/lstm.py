import numpy as np

wfh, wfx, wih, woh, wix, wox = 0, 0, 0, 0, 100, 100
bf, bi, bo, bc, wch, wcx = -100, 100, 0, 0, -100, 50

inp = [0, 0, 1, 1, 1, 0]

h_prev, c_prev = 0, 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(.5 * (1 - sigmoid(-1.15)) ** 2)

# h_history = []

# for x in inp:
#     f = sigmoid(wfh * h_prev + wfx * x + bf)
#     i = sigmoid(wih * h_prev + wix * x + bi)
#     o = sigmoid(woh * h_prev + wox * x + bo)
#     c = f * c_prev + i * np.tanh(wch * h_prev + wcx * x + bc)
#     h = o * np.tanh(c)
#     h_history.append(h)
#     c_prev = c
#     h_prev = h

# print(h_history)

# [0, 0, 1, 1, 1, 0]
# [0, 0, 1, -1, 1, 0]

# [1, 1, 0, 1, 1]
# [1, -1, 0, 1, -1]