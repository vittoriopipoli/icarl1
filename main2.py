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

# from data_loader_2 import iCIFAR10, iCIFAR100
from data_loader import iCIFAR10, iCIFAR100
from model import iCaRLNet


import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

def show_images(images):
    N = images.shape[0]
    fig = plt.figure(figsize=(1, N))
    gs = gridspec.GridSpec(1, N)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.show()


# Hyper Parameters
total_classes = 100
num_classes = 10



transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#function to filter trainset data
def filter(data, classes):
    batch = []
    val = []
    i = 0
    for images, labels in data:
        if labels in classes:
            batch.append([images, labels])
    return batch

#function to check batch's classes
def checkBatchClasses(batch):
    setlab = set()
    for images, labels in batch1:
        # print(labels)
        if labels not in setlab:
            setlab.add(labels)
    print(setlab)
    print(len(setlab))

def getTrainVal(batch):
    curr = batch
    random.shuffle(curr)
    limit = int((len(curr) * 2) / 3)
    train = range(0, limit)
    val = range(limit, len(curr))
    return train, val

def retrieveIndexTrainVal(batch):
    samples = range(0,len(batch))
    labels = []
    for i in samples:
        labels.append(batch[i][1])
    train, val, y_train, y_val = train_test_split(samples,labels,test_size=0.1,
    random_state=42,stratify=labels)
    index_train = []
    index_val = []
    for i, el in enumerate(samples):
        if el in train:
            index_train.append(i)
        else:
            index_val.append(i)
    print("retrieveTrainVal")
    print(len(index_train))
    print(len(index_val))
    return index_train, index_val

def retrieveDataTrainVal(batch, ind_train, ind_val):
    train_dataset = Subset(batch, ind_train)
    val_dataset = Subset(batch, ind_val)
    return train_dataset, val_dataset



# if __name__ == "__main__":


batch1 = filter(trainset, list(range(10, 20)))
print("Batch1 size: {}".format(len(batch1)))
checkBatchClasses(batch1)

ind_train, ind_val = retrieveIndexTrainVal(batch1)
print("ind_train size: {}".format(len(ind_train)))
print("ind_val size: {}".format(len(ind_val)))

batch_train_data, batch_val_data = retrieveDataTrainVal(batch1, ind_train, ind_val)


batch1_loader = torch.utils.data.DataLoader(batch1, batch_size=4,
                                          shuffle=True, num_workers=0)

train_batch1_loader = torch.utils.data.DataLoader(batch_train_data, batch_size=4,
                                          shuffle=True, num_workers=0)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

# # get some random training images
# dataiter = iter(batch1_loader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
#
# dataiter = iter(train_batch1_loader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
# #
# # images, labels = train_batch1_loader.dataset.__getitem__(80)
# # imshow(torchvision.utils.make_grid(images))
#
# images, labels = batch1_loader.dataset.__getitem__(5)
# imshow(torchvision.utils.make_grid(images))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
print(DEVICE)
# Initialize CNN
K = 2000 # total number of exemplars
icarl = iCaRLNet(2048, 1)
# icarl.cuda()
icarl = icarl.to(DEVICE)


for s in range(0, total_classes, num_classes):
    # Load Datasets
    print ("Loading training examples for classes", range(s, s+num_classes))

    batch = filter(trainset, list(range(s, s+num_classes)))
    ind_train, ind_val = retrieveIndexTrainVal(batch1)
    train_set, batch_val_data = retrieveDataTrainVal(batch1, ind_train, ind_val)
    print("Batch1 size: {}".format(len(batch1)))
    print("ind_train size: {}".format(len(ind_train)))
    print("ind_val size: {}".format(len(ind_val)))

    train_loader = torch.utils.data.DataLoader(batch_train_data, batch_size=100,
                                               shuffle=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(batch_val_data, batch_size=100,
                                               shuffle=True, num_workers=0)

    reprdata = []
    for i in range(0,len(ind_train)):
        row = []
        row.append(ind_train[i])        #index
        row.append(train_set[i][0])     #image
        row.append(train_set[i][1])     #label
        reprdata.append(row)

    # Update representation via BackProp
    icarl.update_representation(train_set, reprdata)
    m = K / icarl.n_classes

    # Reduce exemplar sets for known classes
    icarl.reduce_exemplar_sets(m)

    # Construct exemplar sets for new classes
    for y in range(icarl.n_known, icarl.n_classes):
        print ("Constructing exemplar set for class-%d..." %(y))
        images = train_set.get_image_class(y)
        icarl.construct_exemplar_set(images, m, transform_test)
        print ("Done")

    for y, P_y in enumerate(icarl.exemplar_sets):
        print ("Exemplar set for class-%d:" % (y), P_y.shape)
        #show_images(P_y[:10])

    icarl.n_known = icarl.n_classes
    print ("iCaRL classes: %d" % icarl.n_known)

    total = 0.0
    correct = 0.0
    for indices, images, labels in train_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()

    print('Train Accuracy: %d %%' % (100 * correct / total))

    total = 0.0
    correct = 0.0
    for indices, images, labels in test_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()

    print('Test Accuracy: %d %%' % (100 * correct / total))


