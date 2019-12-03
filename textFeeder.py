from feeder import Feeder
from SDR import SDR


class TXTFeeder(Feeder):
    def __init__(self,
                 inputFileName,
                 numBits=512,
                 numOnBits=10,
                 seed=37,
                 ):
        Feeder.__init__(self, numBits, numOnBits)
        self.inputFileName = inputFileName
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
        #return inputSDR

    def numberOfLetter(self):
        file = open(self.inputFileName, "r")
        data = file.read().replace(" ", "")
        return len(data)


    def total_character(self):
        file = open(self.inputFileName, "r")
        data = file.read()
        return len(data)

