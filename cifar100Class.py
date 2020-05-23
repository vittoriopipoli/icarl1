# This class returns a list:
# list[0] = batch of first 10 classes (0, 9) (5000 elements)
# list[1] = batch of second 10 classes (10, 19) (5000 elements)
# ...

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import sys

ROOT = './data'


# This class handles cifar-100 dataset.
# The constructor downloads the data from torchvision.dataset repository
# The method get_batches splits the data into batches of the specified size
# Trainset  50k elements
# Testset   10k elements

class cifar_100:

    def __init__(self, classes_per_batch):

        self.__classes_per_batch = classes_per_batch

        self.__transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # maybe update these values with cifar info
        ])

        self.__transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.__trainset = torchvision.datasets.CIFAR100(root=ROOT, train=True,
                                                        download=True,
                                                        transform=self.__transform_train)  # prima transform = transform ma transform non esiste

        self.__testset = torchvision.datasets.CIFAR100(root=ROOT, train=False,
                                                       download=True, transform=self.__transform_test)  # vedi sopra

        self.__num_classes = 100
        self.__indexes = []

    def get_batches(self, trainset=True):
        batches = []

        if trainset:
            data = self.__trainset
        else:
            data = self.__testset

        labels = np.array(data.targets)
        num_elem = len(labels)

        ids = [False for i in range(num_elem)]
        for c in range(self.__num_classes + 1):
            if c != 0 and not c % self.__classes_per_batch:
                new_ids = []
                for i, v in enumerate(ids):
                    if v:
                        new_ids.append(i)
                ids = new_ids
                batch = Subset(data, ids)
                batches.append(batch)
                if trainset:
                    self.__indexes.append(ids)

                if c == 100:
                    break
                ids = [False for i in range(num_elem)]

            indexes = labels == c
            ids = [i or j for i, j in zip(ids, indexes)]

        return batches

    def get_indexes_of_batch(self, batch_num):

        if len(self.__indexes) <= 1:
            print("ERROR: You first need to get batches for the training set: 'get_batches(trainset=True)'",
                  file=sys.stderr)
            return
        return self.__indexes[batch_num]

    def get_tot_classes_count(self):
        return self.__num_classes

    def get_classes_per_batch(self):
        return self.__classes_per_batch

    def train_val_split(self, val_size):
        train = 0
        val = 0

    def get_transform_test(self):
        return self.__transform_test

    # print("######################## MUST BE IMLPEMENTED! DONT USE IT YET! #############################")

    # return train, val

    def get_first_k_batches(self, k, batches):
        k_max = self.__num_classes / self.__classes_per_batch
        if k > k_max or k < 0:
            print("ERROR: K must be a positive integer from 1 to", int(k_max), file=sys.stderr)
            return None

        first_k = []
        for i in range(k):
            first_k.extend(batches[i])

        return first_k
