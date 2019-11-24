import numpy as np
import random




class Segment(object):
    def __init__(self,
                 seed=37,
                 numColumn=512,
                 cellsPerColumn=32):
        np.random.seed(seed)
        self.numColumn = numColumn
        self.cellsPerColumn = cellsPerColumn


    def getRandomPermance(self, column):
        stdPermanence = 0.20
        weights = np.random.randn(self.numColumn, self.cellsPerColumn) * stdPermanence
        weights[column, :] = -1
        weights = np.minimum(weights, 1)

        return weights

    # def Rand(start, end, num):
    #     res = []
    #     for j in range(num):
    #         res.append(random.randint(start, end))
    #
    #     return res


    def initDistalSegment(self, row, column, potentialSynapses = 40):
        randomCell = [random.randint(0, ((self.numColumn * self.cellsPerColumn) - 1)) for i in range(potentialSynapses)]
        initializedDistalSegment = np.zeros((self.cellsPerColumn, self.numColumn))
        for rCell in randomCell:
            r = rCell / self.numColumn
            c = rCell % self.numColumn
            initializedDistalSegment[r][c] = random.uniform(0, 1)

        return  initializedDistalSegment





s = Segment()
w = s.initDistalSegment(106, 40)
print(w.shape)
count = 0
for i in range(0,32):
    for j in range(0,512):
        if w[i][j] > 0 :
            count += 0.5
print(count)

