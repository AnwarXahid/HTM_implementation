from neuron import Neuron


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
        self.cellularLayer = [self.cellsPerColumn][self.numColumn]


        for i in range(self.cellsPerColumn):
            for j in range(self.numColumn):
                self.cellularLayer[i][j] = Neuron(i,
                                                  j,
                                                  segmentsPerCell=self.segmentsPerCell,
                                                  activationThreshold=self.activationThreshold,
                                                  synapticThreshold=self.synapticThreshold,
                                                  discountFactor=self.discountFactor)

        # randomModule = RandomModule(seed=seed,
        #                             numColumn=numColumn,
        #                             cellsPerColumn=cellsPerColumn)
        # self.numColumn = numColumn
        # self.columns = []
        # self.burstedColumns = np.full(numColumn, False)
        # self.predictedCells = np.full((numColumn, cellsPerColumn), False)
        # self.activatedCells = np.full((numColumn, cellsPerColumn), False)
        # self.previousPredictedCells = np.full((numColumn, cellsPerColumn), False)
        # self.previousActivatedCells = np.full((numColumn, cellsPerColumn), False)
        #
        # self.updateWeight = updateWeight
        # self.predictedColumns = np.full(numColumn, False)
        #
        # self.iteration = 0
        # self.results = []



