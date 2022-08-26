import numpy as np

"""
x1 =[-1,-1],	y1=1
x2 =[1,0],	y2=-1
x3 =[-1, 1.5],	y3=1
"""

xs = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
ys = np.array([1, 1, 1])



print(xs.shape)
print(xs[0].shape)

def perceptron(xs, ys, T):
    w = np.zeros(xs[0].shape)
    for t in range(T):
        for i in range(len(xs)):
            print(f"t, i = {t, i}, w = {w}, ABOUT TO TRY {{xs, ys}} = {xs[i]}, y = {ys[i]}")

            if ys[i] * np.dot(w, xs[i]) <= 0:
                w = w + xs[i]*ys[i]
                print(f"MISTAKE, now w = {w}")
    print(f"w = {w}")
    return w




# if we want to Start with x(2) ie xs[1]
# xs = np.array([*xs[1:], xs[0]])
# ys = np.array([*ys[1:], ys[0]])
print(xs, '\n', ys)


for T in range(1, 2):
    print(f"\t\tYO LOOK | T = {T} | NOW WE RESTART")
    perceptron(xs, ys, T)

theta = np.array([0, 0])
