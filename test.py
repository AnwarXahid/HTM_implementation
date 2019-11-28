from hierarchicalTemporalMemory import hierarchicalTemporalMemory
from textFeeder import TXTFeeder



test_feeder = TXTFeeder("feed_me.txt")
htm = hierarchicalTemporalMemory(feeder=test_feeder)
a = htm.activateNeurons()
#print(htm.activatedNurons.count_nonzero())