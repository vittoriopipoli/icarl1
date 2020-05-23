import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import iCaRLNet
from cifar100Class import cifar_100

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

torch.cuda.current_device()
torch.cuda._initialized = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


# Hyper Parameters
total_classes = 100
num_classes = 10

transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



# Initialize CNN
K = 2000  # total number of exemplars
icarl = iCaRLNet(2048, 0)
# icarl.cuda()
icarl = icarl.to(DEVICE)
BATCH_SIZE = 100
acc_vect = []

cifar = cifar_100(num_classes)
batches = cifar.get_batches(trainset=True)
testset = cifar.get_batches(trainset=False)
test_set = []

for s, (train_batch, test_batch) in enumerate(zip(batches, testset)):
# for s in range(0, num_classes):
    # Load Datasets
    print("Loading training examples for classes", range(s*num_classes, s*num_classes + num_classes))

    # batch, index = filterAndIndex(trainset, list(range(s, s + num_classes)))
    batch = train_batch
    # ind_train, ind_val = retrieveIndexTrainVal(batch)
    # train_set, batch_val_data = retrieveDataTrainVal(batch, ind_train, ind_val)
    print("Batch size: {}".format(len(batch)))
    # print("Index size: {}".format(len(index)))
    # print("ind_train size: {}".format(len(ind_train)))
    # print("ind_val size: {}".format(len(ind_val)))
    test_set.extend(test_batch)
    # test_batch = filter(testset, list(range(0, s + num_classes)))

    train_loader = torch.utils.data.DataLoader(batch, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=0, drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=0, drop_last=True)

    # reprdata = []
    # for i in range(0, len(batch)):
    #     # row = []
    #     # row.append([ind_train[i], train_set[i][0], train_set[i][1]])
    #     # row.append(ind_train[i])        #index
    #     # row.append(train_set[i][0])     #image
    #     # row.append(train_set[i][1])     #label
    #     # reprdata.append(row)
    #     reprdata.append([index[i], batch[i][0], batch[i][1]])

    # Update representation via BackProp
    icarl.update_representation(batch)
    m = K / icarl.n_classes

    # Reduce exemplar sets for known classes
    icarl.reduce_exemplar_sets(m)

    # Construct exemplar sets for new classes
    for y in range(icarl.n_known, icarl.n_classes):
        print("Constructing exemplar set for class-%d..." % (y))
        images = [row[0] for row in batch]
        icarl.construct_exemplar_set(images, m)
        print("Done exemplars")

    # for y, P_y in enumerate(icarl.exemplar_sets):
    #     print ("Exemplar set for class-%d:" % (y), P_y.shape)
    #     #show_images(P_y[:10])

    icarl.n_known = icarl.n_classes
    print("iCaRL classes: %d" % icarl.n_known)

    total = 0.0
    correct = 0.0
    print(len(train_loader))
    for images, labels in train_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform_test)
        # total += labels.size(0)
        total = total + len(labels)
        correct += (preds.data.cpu() == labels).sum()

    print('Train Accuracy: %d %%' % (100 * correct / total))

    total = 0.0
    correct = 0.0

    for images, labels in test_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform_test)
        # total += labels.size(0)
        total = total + len(labels)
        correct += (preds.data.cpu() == labels).sum()

    print('Test Accuracy: %d %%' % (100 * correct / total))
    acc_vect.append(100 * correct / total)

print(acc_vect)
