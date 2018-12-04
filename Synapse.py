import numpy as np
import math

memDecay = 7
threshold = 20
synapseMaxDelay = 8
synapseMinDelay = 1


class Synapse:
    def __init__(self, syn_terminals, syn_num):
        global synapseMaxDelay, synapseMinDelay, memDecay, threshold

# this is used to guarantee each neuron only generate one spike
#        w_min = threshold / (syn_terminals * syn_num * self.lif(synapseMaxDelay))
#        w_max = threshold / (syn_terminals * syn_num * self.lif(synapseMinDelay))
#        w_max = 2
#        w_min = 0
#        self.delays = np.random.uniform(synapseMinDelay, synapseMaxDelay, syn_terminals)
#        self.weights = np.random.uniform(w_min, w_max, syn_terminals)
        self.delays = np.arange(synapseMinDelay, synapseMinDelay + syn_terminals)
        self.weights = np.random.normal(1.5, 1, syn_terminals)
#        self.weights = np.ones(syn_terminals) * w_max


    @classmethod
    def lif(cls, time):
        global memDecay

# PSP = t/tau * exp(1-t/tau)
        if time >= 0:
            div = float(time) / memDecay
            return div * math.exp(1 - div)
        else:
            return 0


