import numpy as np
import LoadSamples as LS
from Network import *


def simulate(samples_train, labels_train, samples_test, labels_test, network_size, inhib_num):
    syn_terminals = 16
    layout = np.asarray(network_size)
    input_num = samples_train.shape
    vth = 20
    network = Network(layout, input_num[1], syn_terminals, inhib_num, len(labels_train[0]))
    Neuron.set_threshold(vth)
    epochs = 500
    learning_rate = 0.01
    accuracy = network.train(samples_train, labels_train, samples_test, labels_test, learning_rate, epochs)
    return accuracy


def main():
# input and output are determined by data
    print("please enter a number to choose different experiments:\n")
    print("1. XOR; 2. iris; 3. Wisconsin Breast cancer dataset\n")
    dataset = input('Choose dataset: ')
    if dataset == 1:
        samples, labels = LS.load_xor()
        network_size = [5]
        inhib_num = 1
        simulate(samples, labels, samples, labels, network_size, inhib_num)
    elif dataset == 2:
        samples_1, labels_1, samples_2, labels_2 = LS.load_iris()
        network_size = [10]
        inhib_num = 2
        a_1 = simulate(samples_1, labels_1, samples_2, labels_2, network_size, inhib_num)
        a_2 = simulate(samples_2, labels_2, samples_1, labels_1, network_size, inhib_num)
        accuracy_1 = np.max(a_1)
        accuracy_2 = np.max(a_2)
        np.savetxt("accuracy_iris_1.txt", accuracy_1)
        np.savetxt("accuracy_iris_2.txt", accuracy_2)
        accuracy = (accuracy_1 + accuracy_2)/2
        print 'CV accuracy: %0.2f%%' % accuracy
    elif dataset == 3:
        samples_1, labels_1, samples_2, labels_2 = LS.load_wbcd()
        network_size = [15]
        inhib_num = 3
        a_1 = simulate(samples_1, labels_1, samples_2, labels_2, network_size, inhib_num)
        a_2 = simulate(samples_2, labels_2, samples_1, labels_1, network_size, inhib_num)
        accuracy_1 = np.max(a_1)
        accuracy_2 = np.max(a_2)
        np.savetxt("accuracy_wbcd_1.txt", accuracy_1)
        np.savetxt("accuracy_wbcd_2.txt", accuracy_2)
        accuracy = (accuracy_1 + accuracy_2)/2
        print 'CV accuracy: %0.2f%%' % accuracy
    else:
        print("please enter a number to choose different experiments:\n")
        print("1. XOR; 2. iris; 3. Wisconsin Breast cancer dataset\n")
        assert()


if __name__ == "__main__":
    main()
