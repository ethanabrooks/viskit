#! /usr/bin/env python

import h5py
import numpy as np
import csv
from tqdm import tqdm
import pickle

dim = 1000
N = int(1e6)
new_data = np.random.random((1, dim))


def hfpy_append():
    with h5py.File("test.h5py", mode="a") as h5f:
        dset = h5f["table"]
        i, j = dset.shape
        dset.resize((i + 1, j))
        dset[i] = new_data


def csv_append():
    with open("test.csv", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(new_data))


if __name__ == "__main__":

    with h5py.File("test.h5py", mode="w") as h5f:
        h5f.create_dataset(
            "table",
            shape=(N, dim),
            maxshape=(None, dim),
            dtype=float,
            compression="gzip",
            chunks=(1, dim),
        )
        h5f["table"][:] = new_data

    with open("test.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for _ in tqdm(range(N)):
            writer.writerow(list(np.random.random((1, dim))))

    with open("test.pkl", "wb") as f:
        pickle.dump(np.random.random((N, dim)), f)
