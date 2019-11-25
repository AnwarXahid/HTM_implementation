import random


class SDR(object):
    def __init__(self,
                 input_list,
                 numBits=512,
                 numOnBits=10,
                 seed=37):

        random.seed(seed)
        population = range(0,numBits-1)
        self.sdr_dict = {i:random.sample(population, numOnBits) for i in input_list}


    def getSDR(self, input):
        return self.sdr_dict[input]