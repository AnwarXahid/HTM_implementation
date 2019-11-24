import numpy as np
import random



DEBUG = True



class Feeder(object):

    def __init__(self,
                 numBits=512,
                 numOnBits=10,
                 seed=42):
        self.numBits = numBits
        self.numOnBits = numOnBits











class SDR(object):

    def __init__(self,
                 input_list,
                 numBits=512,
                 numOnBits=10,
                 seed=42):

        random.seed(seed)
        population = range(0,numBits-1)
        self.sdr_dict = {i:random.sample(population, numOnBits) for i in input_list}

    def getSDR(self, input):
        return self.sdr_dict[input]

    def getInput(self, sdr):
        return 0

    def getCollisionProb(self, n, a, s, theta):
        numerator = 0
        for b in range(theta, s+1):
            numerator += combinatorial(s, b) * combinatorial(n-s, a-b)

        denominator = combinatorial(n, a)

        return numerator*1.0/denominator

def combinatorial(a,b):
    return factorial(a)*1.0/factorial(a-b)/factorial(a)

def factorial(a):
    if a == 1:
        return 1
    else:
        return a*factorial(a-1)









class TXTFeeder(Feeder):

    def __init__(self,
                 inputFileName,
                 numBits=512,
                 numOnBits=10,
                 seed=42,
                 ):
        Feeder.__init__(self, numBits, numOnBits)
        self.char_list = [char for char in open(inputFileName).read()]
        asc_chars = [chr(i) for i in range(128)]
        self.char_sdr = SDR(asc_chars,
                            numBits=numBits,
                            numOnBits=numOnBits,
                            seed=seed)
        self.readIndex = -1

    def feed(self):
        if self.readIndex < len(self.char_list)-1:
            self.readIndex = self.readIndex + 1
        else:
            self.readIndex = -1
        inputChar = self.char_list[self.readIndex]
        inputSDR = self.char_sdr.getSDR(inputChar)
        return inputChar, inputSDR

    def evaluatePrediction(self, inputChar, prediction):
        scores = [(i, self.getMatch(i, prediction)) for i in range(128)]
        # scores = [s for s in scores if s[1] > 4]
        scores.sort(key=lambda x: x[1],reverse=True)
        # print("Input: ", inputChar)
        predChars = ""
        hit = False
        for i, score in scores[:5]:
            newChar = chr(i) + " : " + str(score) + " , "
            predChars += newChar
            if inputChar == chr(i):
                hit = True
        print("Input: ", inputChar, "Prediction: ", predChars)
        if hit:
            return 1.0
        else:
            return 0.0

    def getMatch(self, i, prediction):
        return np.sum(prediction[self.char_sdr.getSDR(chr(i))].astype(np.int))





############ Testing############
temp = TXTFeeder("feed_me.txt")
out1, out2 = temp.feed()
print(out1)
print("\n \n \n \n")
print(out2)
print(random.uniform(0, 1))