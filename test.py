from hierarchicalTemporalMemory import hierarchicalTemporalMemory
from textFeeder import TXTFeeder



test_feeder = TXTFeeder("feed_me.txt")
htm = hierarchicalTemporalMemory(feeder=test_feeder)
htm.runModel()
p = htm.performance()
print("Prediction Performance: " + str(p))