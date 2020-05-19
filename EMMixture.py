import math
import statistics

import numpy as np
import matplotlib.pyplot as plt


class NewtonRaphsonEM:
    def __init__(self, a, b):
        self.alphaInit = a
        self.betaInit = b

    def getHessianEM(self, a, b, x, pykGivenX):
        secAlpha = 0
        n = len(x)
        for i in range(n):
            secAlpha = secAlpha + (math.exp(-(x[i] - a) / b) * pykGivenX[0][i])

        secAlpha = -(1 / b ** 2) * secAlpha

        firstTermAB = 0
        secondTermAB = 0
        for i in range(n):
            firstTermAB = firstTermAB + (math.exp(-(x[i] - a) / b) * pykGivenX[0][i])
            secondTermAB = secondTermAB + ((x[i] - a) * (math.exp(-(x[i] - a) / b)) * pykGivenX[0][i])

        secAlphaBeta = -(np.sum(pykGivenX[0]) / b ** 2) + ((1 / b ** 2) * firstTermAB) - (1 / b ** 3)

        firstTermB = 0
        secondTermB = 0
        thirdTermB = 0
        for i in range(n):
            firstTermB = firstTermB + (x[i] - a)
            secondTermB = secondTermB + ((x[i] - a) * (math.exp(-(x[i] - a) / b)) * pykGivenX[0][i])
            thirdTermB = thirdTermB + ((x[i] - a) ** 2) * (math.exp(-(x[i] - a) / b)) * pykGivenX[0][i]

        secBeta = (np.sum(pykGivenX[0]) / b ** 2) - ((2 / b ** 3) * firstTermB) + ((2 / b ** 3) * secondTermB) - (
                (1 / b ** 4) * thirdTermB)

        hessian = np.array([[secAlpha, secAlphaBeta], [secAlphaBeta, secBeta]])
        return hessian

    def gradientEM(self, a, b, x, pykGivenX):
        n = len(x)
        firstAlpha = 0
        for i in range(n):
            firstAlpha = firstAlpha + (math.exp(-(x[i] - a) / b) * pykGivenX[0][i])

        firstAlpha = (n / b) - ((1 / b) * firstAlpha)

        firstTermB = 0
        secondTermB = 0
        for i in range(n):
            firstTermB = firstTermB + ((x[i] - a) * pykGivenX[0][i])
            secondTermB = secondTermB + ((x[i] - a) * (math.exp(-(x[i] - a) / b)) * pykGivenX[0][i])

        firstBeta = -(np.sum(pykGivenX[0]) / b) + ((1 / b ** 2) * firstTermB) - ((1 / b ** 2) * secondTermB)
        gradient = np.array([[firstAlpha], [firstBeta]])
        return gradient


def pyOfkGivenX(w, a, b, mu, si, data):
    pykGivenX = np.zeros([len(w), len(data)])
    i = 0
    for d in data:
        pxgivenab = (1 / b) * np.exp(-(d - a) / b) * np.exp(-np.exp(-(d - a) / b))
        py1x = w[0] * pxgivenab
        pxgivenms = 1 / (si * np.sqrt(2 * np.pi)) * np.exp(- (d - mu) ** 2 / (2 * si ** 2))
        py2x = w[1] * pxgivenms
        pykGivenX[0][i] = py1x / (py1x + py2x)
        pykGivenX[1][i] = py2x / (py1x + py2x)
        i += 1
    return pykGivenX


def predictWeight(pykGivenX):
    w1 = np.sum(pykGivenX[0]) / len(pykGivenX[0])
    w2 = np.sum(pykGivenX[1]) / len(pykGivenX[1])

    return [w1, w2]


def predictAB(NR, alphaOld, betaOld, data, pykGivenX):
    dif = 9999
    updatedParamab = [alphaOld, betaOld]
    while dif > 0.001:
        h = NR.getHessianEM(alphaOld, betaOld, data, pykGivenX)
        g = NR.gradientEM(alphaOld, betaOld, data, pykGivenX)
        updatedParamab = np.array([[alphaOld], [betaOld]]) - np.matmul(np.linalg.inv(h), g)
        # print(updatedParam)
        dif = updatedParamab - np.array([[alphaOld], [betaOld]])
        dif = np.matmul(dif.transpose(), dif)
        alphaOld = updatedParamab[0][0]
        betaOld = updatedParamab[1][0]
    return updatedParamab


def predictMS(meuOld, sigmaOld, data, pykGivenX):
    Nk = 0
    i = 0
    for d in data:
        Nk = Nk + pykGivenX[1][i]
        i += 1

    mu = 0
    i = 0
    for d in data:
        mu = mu + d * pykGivenX[1][i]
        i += 1
    mu = mu / Nk

    si = 0
    i = 0
    for d in data:
        si = si + (d - mu) ** 2 * pykGivenX[1][i]
        i += 1
    si = si / Nk
    return [mu, si]


#  Generating mixture data
n = [100, 1000]
param = {
    100: [],
    1000: [],
    10000: []
}
# original parameters
alpha, beta = 350, 150
meu, sigma = 600, 50
w1, w2 = 0.5, 0.5

for i in n:
    for j in range(1, 10):
        distributions = [
            {"type": np.random.gumbel, "kwargs": {"loc": alpha, "scale": beta}},
            {"type": np.random.normal, "kwargs": {"loc": meu, "scale": sigma}}
        ]
        coefficients = np.array([w1, w2])
        coefficients /= coefficients.sum()  # in case these did not add up to 1
        sample_size = i

        num_distr = len(distributions)
        data = np.zeros((sample_size, num_distr))
        for idx, distr in enumerate(distributions):
            data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
        random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
        data = data[np.arange(sample_size), random_idx]
        plt.hist(data, bins=100, density=True)
        plt.show()

        alphaOld = 300
        betaOld = 100
        meuOld, sigmaOld = 450, 10
        w = [0.5, 0.5]
        dif = 9999

        NR = NewtonRaphsonEM(alphaOld, betaOld)
        while dif > 0.001:
            pykGivenX = pyOfkGivenX(w, alphaOld, betaOld, meuOld, sigmaOld, data)
            wNew = predictWeight(pykGivenX)
            #  print("wnew: " + str(wNew))
            updatedParamab = predictAB(NR, alphaOld, betaOld, data, pykGivenX)
            #  print("ab: " + str(updatedParamab))
            updatedParamMS = predictMS(meuOld, sigmaOld, data, pykGivenX)
            #  print("ms: " + str(updatedParamMS))
            allParam = np.array([wNew[0], wNew[1], updatedParamab[0][0], updatedParamab[1][0], updatedParamMS[0], updatedParamMS[1]])
            dif = allParam - np.array([w[0], w[1], alphaOld, betaOld, meuOld, sigma])
            dif = np.matmul(dif.transpose(), dif)
            meuOld = updatedParamMS[0]
            sigmaOld = updatedParamMS[1]
            alphaOld = updatedParamab[0][0]
            betaOld = updatedParamab[1][0]
            w = wNew
        param[i].append((w, alphaOld, betaOld, meuOld, sigmaOld))
        print(param)

alphaMeanSD = {
    100: [],
    1000: [],
    10000: []
}
betaMeanSD = {
    100: [],
    1000: [],
    10000: []
}
wMeanSD = {
    100: [],
    1000: [],
    10000: []
}
muMeanSD = {
    100: [],
    1000: [],
    10000: []
}


# for i in n:
#     alphaMeanSD[i] = (statistics.mean(al[0] for al in param[i]), statistics.stdev(al[0] for al in param[i]))
#     betaMeanSD[i] = (statistics.mean(be[1] for be in param[i]), statistics.stdev(be[1] for be in param[i]))
#
# print("Original parameters:")
# print("alpha: " + str(alpha) + " beta: " + str(beta))
# print(" ")
#
# for i in n:
#     print("For n=" + str(i) + ":")
#     print("Alpha:")
#     print("Mean: " + str(round(alphaMeanSD[i][0], 2)) + " Standard Deviation:" + str(round(alphaMeanSD[i][1], 2)))
#     print("Beta:")
#     print("Mean: " + str(round(betaMeanSD[i][0], 2)) + " Standard Deviation:" + str(round(betaMeanSD[i][1], 2)))
#     print(" ")
