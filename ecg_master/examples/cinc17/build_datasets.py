import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

#cj debug
import matplotlib.pyplot as plt


STEP = 256 #300Hz采样的数据 分成每小段0.85s,如是200Hz采样的数据，分成每小段1.28s

def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()

def load_all(data_path):
    label_file = os.path.join(data_path, "../REFERENCE-v3.csv")
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
    for record, label in tqdm.tqdm(records):
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)
        
        plt.plot(ecg)
        plt.pause(1)
        
        num_labels = int(ecg.shape[0] / STEP) #cj modify add int()
        dataset.append((ecg_file, [label]*num_labels))
    return dataset 

def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')

if __name__ == "__main__":
    random.seed(2018)

    dev_frac = 0.1
    data_path = "data/training2017/"
    dataset = load_all(data_path)
    train, dev = split(dataset, dev_frac)
    make_json("train.json", train)
    make_json("dev.json", dev)

