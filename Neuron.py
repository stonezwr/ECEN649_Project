import numpy as np
import math
from Synapse import *

memDecay = 7
threshold = 20
time_end = 40


class Neuron:
    def __init__(self, syn_num, syn_terminals, inhib_num):
        self.fire_time = list()
        self.synapses = np.empty(syn_num, dtype=object)
        if inhib_num > 0:
            self.type = -1
        else:
            self.type = 1
        for s in range(syn_num):
            self.synapses[s] = Synapse(syn_terminals, syn_num)

    @classmethod
    def set_threshold(cls, th):
        global threshold
        threshold = th

    def get_num_terminals(self):
        num_terminals = self.synapses[0].delays.shape
        return num_terminals[0]

    def get_num_synapses(self):
        num_syn = self.synapses.shape
        return num_syn[0]

    def get_last_spike_time(self):
        global time_end
        if self.fire_time:
            return self.fire_time[0]
        else:
            return -1

    def lif_model_differential(self, t, pre_neuron_type):
        global memDecay
        if t > 0:
            div = 1 - (float(t) / memDecay)
            return div * math.exp(div) * pre_neuron_type / memDecay
        else:
            return 0

    def lif_model(self, t, pre_neuron_type):
        global memDecay
        if t > 0:
            div = float(t) / memDecay
            mem = div * pre_neuron_type * math.exp(1 - div)
            return mem
        else:
            return 0

    def error_backprop(self, pre_spike, time, delay, pre_neuron_type):
        t = float(time) - pre_spike - delay
        return self.lif_model_differential(t, pre_neuron_type)

    def post_synaptic_potential(self, pre_spike, time, delay, pre_neuron_type):
        t = float(time) - pre_spike - delay
        psp = self.lif_model(t, pre_neuron_type)
        return psp

    def init_state(self, pre_syn_spike, time, pre_neuron_types):
        syn_num = self.get_num_synapses()
        v_mem = 0.0

        for syn in range(syn_num):
            if time >= pre_syn_spike[syn] >= 0:
                terminals = self.get_num_terminals()
                for t in range(terminals):
                    v_mem += self.synapses[syn].weights[t] \
                        * self.post_synaptic_potential(pre_syn_spike[syn], time, self.synapses[syn].delays[t], pre_neuron_types[syn])

        return v_mem

    def feedforward(self, pre_spike_time, time, pre_neuron_types):
        global threshold
        if len(self.fire_time) == 0:
            v_mem = self.init_state(pre_spike_time, time, pre_neuron_types)
#            if len(self.synapses) == 5:
#                print v_mem
            if v_mem >= threshold:
                self.fire_time.append(time)

    def update_weights(self, delta):
        syn_num = self.get_num_synapses()
        for syn in range(syn_num):
            self.synapses[syn].weights = np.add(self.synapses[syn].weights, delta[syn, :])
            self.synapses[syn].weights[self.synapses[syn].weights < 0] = 0
            if len(self.synapses[syn].weights[self.synapses[syn].weights < 0]) != 0:
                print self.synapses[syn].weights
                assert()

    def reset_neuron(self):
        del self.fire_time
        self.fire_time = list()

    def print_weight(self):
        syn_num = self.get_num_synapses()
        for syn in range(syn_num):
            print self.synapses[syn].weights
