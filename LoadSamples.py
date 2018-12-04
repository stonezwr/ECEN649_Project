import numpy as np
import math


def gaussian(x, mu, sig):
    return 1. / (math.sqrt(2. * math.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)


def gaussian_rf(value, i_min, i_max, m):

    beta = 1.5
    width = 1 * (i_max - i_min) / (beta * (m - 2))
    peak = gaussian(0, 0, width)
    s = []
    for i in range(int(m)):
        if value < 0:
            s.append(-1)
            continue
        center = i_min + (2 * i - 3) * (i_max - i_min) / (2 * (m - 2))
        v = gaussian(value, center, width)/peak
        v = 1 - v
        if v > 0.9:
            s.append(-1)
        else:
            v = math.floor(v / 0.1) + 1
            assert v < 10
            s.append(v)
    return s


def encoding_iris(sample, label):
    s = []
    i_min = 0.0
    i_max = 8.0
    m = 12.0
    for index, item in enumerate(sample):
        s.append(gaussian_rf(item, i_min, i_max, m))
    flat_s = [item for sublist in s for item in sublist]
# append 0 for reference
    flat_s.append(0)
    l = []
    for i in range(3):
        if i == label:
            l.append(15)
        else:
            l.append(21)
    l = np.array(l)
    return flat_s, l


def encoding_wbcd(sample, label):
    s = []
    i_min = 1.0
    i_max = 10.0
    m = 7.0
    for index, item in enumerate(sample):
        s.append(gaussian_rf(item, i_min, i_max, m))
    flat_s = [item for sublist in s for item in sublist]
# append 0 for reference
    flat_s.append(0)
    ll = []
    for i in range(2):
        if i * 2 + 2 == label:
            ll.append(15)
        else:
            ll.append(21)
    return flat_s, ll


def load_xor():
    samples = np.asarray([
        [0, 0, 0]
        ,
        [0, 6, 0]
        , [6, 0, 0]
        , [6, 6, 0]
    ])
    labels = np.asarray([
        [16]
        ,
        [10]
        , [10]
        , [16]
    ])
    return samples, labels


def load_iris():
    a = np.loadtxt('iris.txt', dtype=str)
    samples = []
    labels = []
    label_record = []
    for index, s in enumerate(a):
        s_out = s.split(",")
        s_float = map(eval, s_out[0:4])
        samples.append(s_float)
        if s_out[4] not in label_record:
            label_record.append(s_out[4])
        i = label_record.index(s_out[4])
        labels.append(i)
    sample_encode = []
    label_encode = []
    for index, sample in enumerate(samples):
        s, l = encoding_iris(sample, labels[index])
        sample_encode.append(s)
        label_encode.append(l)
    samples = np.array(sample_encode)
    labels = np.array(label_encode)
    samples_1 = np.concatenate((samples[0:25, :], samples[50:75, :], samples[100:125, :]), axis=0)
    labels_1 = np.concatenate((labels[0:25, :], labels[50:75, :], labels[100:125, :]), axis=0)
    samples_2 = np.concatenate((samples[25:50, :], samples[75:100, :], samples[125:150, :]), axis=0)
    labels_2 = np.concatenate((labels[25:50, :], labels[75:100, :], labels[125:150, :]), axis=0)
    return samples_1, labels_1, samples_2, labels_2


def load_wbcd():
    a = np.loadtxt('breast-cancer-wisconsin.txt', dtype=str)
    labels_1 = []
    samples_1 = []
    labels_2 = []
    samples_2 = []
    malignant = 0
    benign = 0
    for index, s in enumerate(a):
        s_out = s.split(",")
        s_out = np.array(s_out, dtype=int)
        sample = s_out[1:10]
        label = s_out[10:11]
        ss, ll = encoding_wbcd(sample, label)
        if label == 2:
            if benign < 229:
                labels_1.append(ll)
                samples_1.append(ss)
            else:
                labels_2.append(ll)
                samples_2.append(ss)
            benign += 1
        else:
            if malignant < 120:
                labels_1.append(ll)
                samples_1.append(ss)
            else:
                labels_2.append(ll)
                samples_2.append(ss)
            malignant += 1
    samples_1 = np.array(samples_1)
    labels_1 = np.array(labels_1)
    samples_2 = np.array(samples_2)
    labels_2 = np.array(labels_2)
    return samples_1, labels_1, samples_2, labels_2

