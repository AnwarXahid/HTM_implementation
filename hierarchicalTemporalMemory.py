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
        self.inputChar, self.inputSDR = self.feeder.feed()
        #return self.inputChar, self.inputSDR


    def activateNeurons(self):
        flag = 0

        for item in self.inputSDR:
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



    def setPreviouslyActivatedNeurons(self):
        self.previouslyActivatedNeurons = self.activatedNurons.copy()


    def setPreviouslyPredictedNeurons(self):
        self.previouslyPredictedNeurons = self.predictedNeurons.copy()


    def havingPrediction(self, column):
        for i in range(self.cellsPerColumn):
            if self.previouslyPredictedNeurons[i, column] == 1:
                return True

        return False


    def updateSegmentsAndSynapses(self):
        for item in self.inputSDR:
            for i in range(self.cellsPerColumn):
                if self.havingPrediction(item):
                        if self.previouslyPredictedNeurons[i, item] == 1 and self.activatedNurons[i, item] == 1:
                            contributionarySegments = self.cellularLayer[i][item].segmentsContributedToActivation(self.previouslyActivatedNeurons)
                            self.reinforce(contributionarySegments, i, item)
                        elif self.previouslyPredictedNeurons[i, item] == 1 and self.activatedNurons[i, item] == 0:
                            contributionarySegments = self.cellularLayer[i][item].segmentsContributedToActivation(self.previouslyActivatedNeurons)
                            self.adjustingFlasePositive(contributionarySegments, i, item)

                else:
                    self.cellularLayer[i][item].updateSegmentHavingMostinput(self.previouslyActivatedNeurons)
                    break



    def reinforce(self, contributionarySegments, row, column):
        for i in range(len(contributionarySegments)):
            self.cellularLayer[row][column].updateSynapses(i, self.previouslyActivatedNeurons)


    def adjustingFlasePositive(self, contributionarySegments, row, column):
        for i in range(len(contributionarySegments)):
            self.cellularLayer[row][column].punish(i, self.previouslyActivatedNeurons)


    def calculatePredictionPerformance(self):
        # posSeg = self.segments[index].getSegmentWithPositiveEntries()
        # posSyn = np.multiply(posSeg.toarray(), state.toarray())
        self.countPerfectPrediction = 0
        nonZeroIndices = self.previouslyPredictedNeurons.nonzero()
        x_indices, y_indices = nonZeroIndices
        for i in range(len(x_indices)):
            if self.activatedNurons[x_indices[i], y_indices[i]] == 0:
                return False

        self.countPerfectPrediction += 1
        return True



    def runModel(self):
        for i in range(self.feeder.total_character()):
            self.feedForward()
            self.activateNeurons()
            self.predictorNeurons()
            self.updateSegmentsAndSynapses()
            self.setPreviouslyActivatedNeurons()
            self.setPreviouslyPredictedNeurons()
            self.calculatePredictionPerformance()


    def performance(self):
        return (self.countPerfectPrediction / self.feeder.numberOfLetter()) * 100






