import numpy as np
import random




class Segment(object):
    def __init__(self,
                 row,
                 column,
                 numColumn=512,
                 cellsPerColumn=32,
                 potentialSynapses=40,
                 seed=37):
        np.random.seed(seed)
        self.row = row
        self.column = column
        self.numColumn = numColumn
        self.cellsPerColumn = cellsPerColumn
        self.potentialSynapses = potentialSynapses
        highestIndex = (self.numColumn * self.cellsPerColumn) - 1
        randomCell = [random.randint(0, highestIndex) for i in range(self.potentialSynapses)]
        self.distalSegment = np.zeros((self.cellsPerColumn, self.numColumn))
        for rCell in randomCell:
            r = rCell // self.numColumn
            c = rCell % self.numColumn
            self.distalSegment[r][c] = random.uniform(0, 1)            #randomly selected permanence value
        self.distalSegment[self.row][self.column] = 0                  #avoiding self-connected synapse


    def getDistalSegment(self):
        return self.distalSegment


    def connectedDistalSegment(self, threshold=0.5):
        conDistalSegment = np.zeros((self.cellsPerColumn, self.numColumn))
        for i in range(self.cellsPerColumn):
            for j in range(self.numColumn):
                if self.distalSegment[i][j] > threshold:
                    conDistalSegment[i][j] = 1

        return conDistalSegment




class Neuron(object):
    def __init__(self,
                 row,
                 column,
                 segmentsPerCell=4,
                 activationThreshold=15,
                 synapticThreshold=0.5,
                 discountFactor=0.1):
        self.row = row
        self.column = column
        self.activationThreshold = activationThreshold
        self.synapticThreshold = synapticThreshold
        self.discountFactor = discountFactor
        self.segments = []

        for i in range(segmentsPerCell):
            self.segments.append(Segment(self.row, self.column))





# s = Segment()
# s.initDistalSegment(15, 400)
# w = s.connectedDistalSegment()
# count = 0
# for i in range(0,32):
#     for j in range(0,512):
#         if w[i][j] > 0 :
#             count += 1
# print(count)
# n = Neuron(12, 106)
# for segment in n.segments:
#     print(segment.connectedDistalSegment(.5))

