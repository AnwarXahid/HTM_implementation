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
        x_indices, y_indices = nonZeroSynapes
        for i in range(len(x_indices)):
            if self.distalSegment[x_indices[i], y_indices[i]] > threshold:
                conDistalSegment[x_indices[i], y_indices[i]] = 1

        return conDistalSegment


    def getSegmentWithPositiveEntries(self):
        posSegment = lil_matrix((self.cellsPerColumn, self.numColumn))
        nonZeroSynapes = self.distalSegment.nonzero()
        x_indices, y_indices = nonZeroSynapes
        for i in range(len(x_indices)):
            if self.distalSegment[x_indices[i], y_indices[i]] > 0:
                posSegment[x_indices[i], y_indices[i]] = 1

        return posSegment




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
            #print(conSyn)
            temp = linalg.norm(conSyn, 1)
            print(temp)
            if temp > self.activationThreshold:
                return True

        return False


    def segmentsContributedToActivation(self, previousState):
        contributionarySegments = []
        for i in range(self.segmentsPerCell):
            conSeg = self.segments[i].connectedDistalSegment(self.synapticThreshold)
            conSyn = np.multiply(conSeg.toarray(), previousState.toarray())
            #print(conSyn)
            temp = linalg.norm(conSyn, 1)
            print(temp)
            if temp > self.activationThreshold:
                contributionarySegments.append(i)

        return contributionarySegments




    def updateSegmentHavingMostinput(self, state):
        flag = float('-inf')
        for i in range(self.segmentsPerCell):
            posSeg = self.segments[i].getSegmentWithPositiveEntries()
            posSyn = np.multiply(posSeg.toarray(), state.toarray())
            temp = linalg.norm(posSyn, 1)

            if temp > flag:
                index = i

        nonZeroSynapes = self.segments[i].distalSegment.nonzero()
        x_indices, y_indices = nonZeroSynapes
        for i in range(len(x_indices)):
            if posSyn[x_indices[i], y_indices[i]] > 0:
                self.segments[index].distalSegment[x_indices[i], y_indices[i]] += self.discountFactor
            else:
                self.segments[index].distalSegment[x_indices[i], y_indices[i]] -= self.discountFactor



    def updateSynapses(self, index, state):
        posSeg = self.segments[index].getSegmentWithPositiveEntries()
        posSyn = np.multiply(posSeg.toarray(), state.toarray())
        nonZeroSynapes = self.segments[index].distalSegment.nonzero()
        x_indices, y_indices = nonZeroSynapes
        for i in range(len(x_indices)):
            if posSyn[x_indices[i], y_indices[i]] > 0:
                self.segments[index].distalSegment[x_indices[i], y_indices[i]] += self.discountFactor
            else:
                self.segments[index].distalSegment[x_indices[i], y_indices[i]] -= self.discountFactor


    def punish(self, index, state):
        posSeg = self.segments[index].getSegmentWithPositiveEntries()
        nonZeroSynapes = self.segments[index].posSeg.nonzero()
        x_indices, y_indices = nonZeroSynapes
        for i in range(len(x_indices)):
                self.segments[index].distalSegment[x_indices[i], y_indices[i]] -= self.discountFactor