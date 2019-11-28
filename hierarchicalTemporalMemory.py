import numpy as np
from neuron import Neuron
from scipy.sparse import lil_matrix


class hierarchicalTemporalMemory(object):
    def __init__(self,
                 cellsPerColumn=8,
                 numColumn=512,
                 segmentsPerCell=32,
                 activationThreshold=100,
                 synapticThreshold=0.5,
                 discountFactor=0.1,
                 feeder=None,
                 seed=37):
        self.feeder = feeder
        self.cellsPerColumn = cellsPerColumn
        self.numColumn = numColumn
        self.segmentsPerCell = segmentsPerCell
        self.activationThreshold = activationThreshold
        self.synapticThreshold = synapticThreshold
        self.discountFactor = discountFactor
        self.cellularLayer = [ [Neuron(j,
                                       i,
                                       numColumn,
                                       cellsPerColumn,
                                       segmentsPerCell=8,
                                       activationThreshold=15,
                                       synapticThreshold=.5,
                                       discountFactor=.1) for i in range(self.numColumn)] for j in range(self.cellsPerColumn) ]

        self.activatedNurons = lil_matrix((cellsPerColumn, numColumn))
        self.previouslyActivatedNeurons = lil_matrix((cellsPerColumn, numColumn))
        self.predictedNeurons = lil_matrix((cellsPerColumn, numColumn))
        self.previouslyPredictedNeurons = lil_matrix((cellsPerColumn, numColumn))



    def feedForward(self):
        inputChar, inputSDR = self.feeder.feed()
        return inputChar, inputSDR


    def activateNeurons(self):
        inputChar, inputSDR = self.feedForward()
        flag = 0

        for item in inputSDR:
            ###########  activate a cell in a winning column if it was previously in a predictive state #########
            for i in range(self.cellsPerColumn):
                if self.previouslyPredictedNeurons[i, item] == 1:
                    self.activatedNurons[i,item] = 1
                    #print(i,item,flag)
                    flag = 1
                    break
            ######## activate all cells in that column #########
            if flag == 0:
                for i in range(self.cellsPerColumn):
                    self.activatedNurons[i, item] = 1
                    #print(i,item,flag)
            else:
                flag = 0




    def predictorNeurons(self):
        for i in range(self.cellsPerColumn):
            for j in range(self.numColumn):
                if self.cellularLayer[i][j].hasActiveSegment(self.activatedNurons) == True:
                    self.predictedNeurons[i, j] = 1





