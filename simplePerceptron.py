import numpy as np
from matplotlib import pyplot as plt

def activacion(x, w):
    y = np.sign(np.dot(x, w))
    return y

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

def sigmActivation(x1, x2, w1, w2, tetha):
    res = sigmoid((x1*w1) + (x2*w2) - tetha)
    return int(res >= 0.5)

def estimacion(z, y):
    e = z - y
    return e

def correccion(x, e, n):
    dw = np.dot(x.transpose(), e)
    dw = dw * n
    return dw

def entrenamiento(w, x, z, n, epsilon = 0, maxIter = 5000, activationFunc = activacion):
    t = 1
    f, c = x.shape
    e = 1
    while (e > epsilon) and (t < maxIter):
        e = 0
        for h in range(0, f):
            y = activationFunc(x, w)
            eh = estimacion(z, y)
            dw = correccion(x, eh, n)
            w = w + dw
            e = e + np.absolute(eh).sum()
        t += 1
    return w, t

print(0)
