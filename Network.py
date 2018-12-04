from Neuron import *
import numpy as np
import math

time_end = 40


class Network:
    def __init__(self, layout, input_num, syn_terminals, inhib_num, output_num):
        layer_num = layout.shape
        self.layers = list()
# for hidden neurons
        for layer in range(layer_num[0]):
            neuron_num = layout[layer]
            self.layers.append(np.empty(neuron_num, dtype=object))
            if layer == 0:
                pre_num = input_num
            else:
                pre_num = layout[layer - 1]
            for n in range(neuron_num):
                self.layers[layer][n] = Neuron(pre_num, syn_terminals, inhib_num)
                inhib_num -= 1
# for output neurons
        self.layers.append(np.empty(output_num, dtype=object))
        syn_num = layout[-1]
        for n in range(output_num):
            self.layers[-1][n] = Neuron(syn_num, syn_terminals, 0)

    @classmethod
    def shuffle_samples(cls, samples, labels):
        samples_size = samples.shape
        index = np.arange(samples_size[0])
        np.random.shuffle(index)
        labels = labels[index, :]
        samples = samples[index, :]
        return samples, labels

    def get_spike_times(self, layer_index):
        layer = self.layers[layer_index]
        neuron_num = layer.shape
        pre_spike_time = np.zeros(neuron_num[0])

        for n in range(neuron_num[0]):
            pre_spike_time[n] = layer[n].get_last_spike_time()
        return pre_spike_time

    def get_types(self, layer_index):
        layer = self.layers[layer_index]
        neuron_num = layer.shape
        pre_type = np.zeros(neuron_num[0])

        for n in range(neuron_num[0]):
            pre_type[n] = layer[n].type
        return pre_type

    def forward(self, pre_layer_index, cur_layer_index, time):
        cur_layer = self.layers[cur_layer_index]

        cur_neuron_num = cur_layer.shape

        if type(pre_layer_index) is not np.ndarray:
            pre_spike_time = self.get_spike_times(pre_layer_index)
            pre_neuron_types = self.get_types(pre_layer_index)
        else:
            pre_spike_time = pre_layer_index
            pre_neuron_types = np.ones(pre_layer_index.size)

        for n in range(cur_neuron_num[0]):
#            if cur_neuron_num[0] == 2:
#                print 'output ', n
            cur_layer[n].feedforward(pre_spike_time, time, pre_neuron_types)

    def get_spikes(self, index):
        global time_end
        layer = self.layers[index]
        neuron_num = layer.shape
        pre_neuron_spikes = np.zeros(neuron_num[0])

        for n in range(neuron_num[0]):
            t = layer[n].get_last_spike_time()
            pre_neuron_spikes[n] = t
        return pre_neuron_spikes

    def feedforward(self, inputs):
        global time_end
        layer_num = len(self.layers)

        time = 0

        while time <= time_end:
            self.forward(inputs, 0, time)

            for layer in range(1, layer_num):
                self.forward(layer - 1, layer, time)
            outputs = self.get_spikes(layer_num - 1)
#            if len(outputs[outputs > 0]) == len(outputs):
#                break
            time += 1

        return outputs

    def delta_output(self, neuron, pre_spike_time, pre_neuron_types, label):
        output = neuron.get_last_spike_time()
        error = label - output

        syn_num = neuron.get_num_synapses()
        terminal_num = neuron.get_num_terminals()

        delta = 0.0

        for syn in range(syn_num):
            if pre_spike_time[syn] != -1:
                for terminal in range(terminal_num):
                    delta += neuron.synapses[syn].weights[terminal] * neuron.error_backprop(pre_spike_time[syn], output, neuron.synapses[syn].delays[terminal], pre_neuron_types[syn])

        return float(error) / delta

    def delta_hidden(self, neuron, pre_spike_time, pre_neuron_types, next_layer_index, delta_next, cur_neuron_index):
        next_error = 0.0
        cur_error = 0.0

        cur_spike = neuron.get_last_spike_time()

        syn_num = neuron.get_num_synapses()
        terminal_num = neuron.get_num_terminals()

        next_layer = self.layers[next_layer_index]
        neuron_num = next_layer.shape

        for syn in range(syn_num):
            if pre_spike_time[syn] != -1:
                for terminal in range(terminal_num):
                    cur_error += neuron.synapses[syn].weights[terminal] * neuron.error_backprop(pre_spike_time[syn], cur_spike, neuron.synapses[syn].delays[terminal], pre_neuron_types[syn])

        for neuron_index in range(neuron_num[0]):
            terminal_num = next_layer[neuron_index].get_num_terminals()
            next_layer_sum = 0.0
            for terminal in range(terminal_num):
# original code
#                next_layer_sum += next_layer[neuron_index].synapses[cur_neuron_index].weights[terminal] * neuron.error_backprop(cur_spike, next_layer[neuron_index].get_last_spike_time(), next_layer[neuron_index].synapses[cur_neuron_index].delays[terminal], neuron.type)
# my interpretation
                next_layer_sum += next_layer[neuron_index].synapses[cur_neuron_index].weights[terminal] * neuron.error_backprop(cur_spike, next_layer[neuron_index].get_last_spike_time(), next_layer[neuron_index].synapses[cur_neuron_index].delays[terminal], neuron.type)

            # print 'delta next layer ', deltaNextLayer[n]
            next_error += next_layer_sum * delta_next[neuron_index]

        return next_error / cur_error

    def backpropagate(self, label, inputs, learning_rate):

        layer_num = len(self.layers)
        out_layer = self.layers[-1]
        neuron_num = out_layer.shape
        assert(neuron_num[0] == len(label))

        delta = np.zeros(neuron_num[0])

# for output neuron
        for neuron_index in range(neuron_num[0]):
            out_neuron = out_layer[neuron_index]
            # assert(out_neuron.get_last_spike_time() != -1)
            if out_neuron.get_last_spike_time() != -1:
                syn_num = out_neuron.get_num_synapses()
                terminal_num = out_neuron.get_num_terminals()

                delta_y = np.zeros((syn_num, terminal_num))

                pre_spike_time = self.get_spikes(layer_num - 2)
                pre_neuron_types = self.get_types(layer_num - 2)

                for syn in range(syn_num):
                    if pre_spike_time[syn] != -1:
                        for terminal in range(terminal_num):
                            y = out_neuron.post_synaptic_potential(pre_spike_time[syn], out_neuron.get_last_spike_time(), out_neuron.synapses[syn].delays[terminal], pre_neuron_types[syn])

                            delta_y[syn, terminal] = y * (-1) * learning_rate
                    else:
                        delta_y[syn] = np.zeros(terminal_num)

# store for reuse for hidden layers
                delta[neuron_index] = self.delta_output(out_neuron, pre_spike_time, pre_neuron_types, label[neuron_index])

                updates = np.multiply(delta_y, delta[neuron_index])

                out_neuron.update_weights(updates)
            else:
                delta[neuron_index] = 0
# for hidden neuron
        delta_pre = delta
        for l in range(- 2, - layer_num - 1, -1):
            neuron_num = self.layers[l].shape
            delta_h = np.zeros(neuron_num[0])
            for neuron_index in range(neuron_num[0]):
                neuron = self.layers[l][neuron_index]
                if neuron.get_last_spike_time() != -1:
                    syn_num = neuron.get_num_synapses()
                    terminal_num = neuron.get_num_terminals()
                    delta_h_y = np.zeros((syn_num, terminal_num))
                    if l > - layer_num:
                        pre_spike_time = self.get_spikes(l - 1)
                        pre_neuron_types = self.get_types(l - 1)
                    else:
                        pre_spike_time = inputs
                        pre_neuron_types = np.ones(inputs.size)
                    for syn in range(syn_num):
                        if pre_spike_time[syn] != -1:
                            for terminal in range(terminal_num):
                                y = neuron.post_synaptic_potential(pre_spike_time[syn], neuron.get_last_spike_time(), neuron.synapses[syn].delays[terminal], pre_neuron_types[syn])
                                delta_h_y[syn, terminal] = y * (-1) * learning_rate
                        else:
                            delta_h_y[syn] = np.zeros(terminal_num)

                    # compute the hidden layer delta
                    delta_h[neuron_index] = self.delta_hidden(neuron, pre_spike_time, pre_neuron_types, l + 1, delta_pre, neuron_index)
                    updates = delta_h_y*delta_h[neuron_index]
                    neuron.update_weights(updates)
                else:
                    delta_h[neuron_index] = 0
            delta_pre = delta_h

    def predict_error(self, label, predict):
        neuron_num = len(label)
        error = 0
        #print label
        #print predict
        for n in range(neuron_num):
            error += math.pow((predict[n] - label[n]), 2)
        return error

    def predict_accuracy(self, label, output):
        label_neuron = np.argmin(label)
        output_neuron = np.argmin(output)
        if label_neuron == output_neuron:
            return 1
        else:
            return 0

    def reset_network(self):
        layer_num = len(self.layers)

        for layer_index in range(layer_num):
            neuron_num = self.layers[layer_index].shape
            for neuron in range(neuron_num[0]):
                self.layers[layer_index][neuron].reset_neuron()

    def train(self, samples_train, labels_train, samples_test, labels_test, learning_rate, epochs):
        best_accuracy = 0
        best_epoch = 0
        all_accuracy = []
        print '****************Start of simulation***************'
        for epoch in range(epochs):
           # if epoch % 100 == 0:
           #     learning_rate = learning_rate/2
            print '************ %d epoch start ***************' % epoch
            # samples, labels = self.shuffle_samples(samples, labels)
            print '************ Training phase ***************'
            sample_num, neuron_index = samples_train.shape
            for sample_index in range(sample_num):
                inputs = samples_train[sample_index, :]
                label = labels_train[sample_index]

                self.feedforward(inputs)

                self.backpropagate(label, inputs, learning_rate)

                self.reset_network()
# testing phase
            print '************ Testing phase ***************'
            sample_num, neuron_index = samples_test.shape
            error = 0
            correct = 0
            for sample_index in range(sample_num):
                inputs = samples_test[sample_index, :]
                label = labels_test[sample_index]
                #print inputs
                output = self.feedforward(inputs)

                error += self.predict_error(label, output)
                correct += self.predict_accuracy(label, output)

                self.reset_network()
            error = error/sample_num
            print 'The error is: ', error
            accuracy = float(correct)*100/float(sample_num)
            print('Accuracy is: %0.2f%%' % accuracy)
            all_accuracy.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
            if error < 1:
                break
        print 'accuracy: %0.2f%% @ epoch: %d' % (best_accuracy, best_epoch)
        print 'output 0:'
        self.layers[-1][-1].print_weight()
        all_accuracy = np.array(all_accuracy)
        return all_accuracy
