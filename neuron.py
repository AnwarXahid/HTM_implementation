import numpy as np
import random
from scipy.sparse import lil_matrix
from scipy import linalg





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
        self.distalSegment = lil_matrix((self.cellsPerColumn, self.numColumn))
        for rCell in randomCell:
            r = rCell // self.numColumn
            c = rCell % self.numColumn
            self.distalSegment[r, c] = random.uniform(0, 1)            #randomly selected permanence value
        self.distalSegment[self.row, self.column] = 0                  #avoiding self-connected synapse




    def getDistalSegment(self):
        return self.distalSegment


    def connectedDistalSegment(self, threshold=0.5):
        conDistalSegment = lil_matrix((self.cellsPerColumn, self.numColumn))
        nonZeroSynapes = self.distalSegment.nonzero()
        for item in nonZeroSynapes:
            x_index, y_index = item
            if self.distalSegment[x_index, y_index] > threshold:
                conDistalSegment[x_index, y_index] = 1

        return conDistalSegment




class Neuron(object):
    def __init__(self,
                 row,
                 column,
                 numColumn=512,
                 cellsPerColumn=32,
                 segmentsPerCell=4,
                 activationThreshold=15,
                 synapticThreshold=0.5,
                 discountFactor=0.1,
                 feeder = None):
        self.row = row
        self.column = column
        self.segmentsPerCell = segmentsPerCell
        self.activationThreshold = activationThreshold
        self.synapticThreshold = synapticThreshold
        self.discountFactor = discountFactor
        self.segments = []

        for i in range(segmentsPerCell):
            self.segments.append(Segment(self.row, self.column, numColumn, cellsPerColumn))




    def hasActiveSegment(self, currentState):
        for i in range(self.segmentsPerCell):
            conSeg = self.segments[i].connectedDistalSegment(self.synapticThreshold)
            conSyn = np.multiply(conSeg.toarray(), currentState.toarray())
            if linalg.norm(conSyn.todense(), ord=1) > self.activationThreshold:
                return True

        return False



