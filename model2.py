import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from icarl1.resnet import resnet18 #inutilizzata per ora
from icarl1.resnet import resnet34 #inutilizzata per ora
from icarl1.resnetNapo import resnet32
import torchvision
import math

# Hyper Parameters
# num_epochs = 50
num_epochs = 50
batch_size = 128
learning_rate = 0.002


class iCaRLNet(nn.Module):
    def __init__(self, feature_size, n_classes):
        # Network architecture
        super(iCaRLNet, self).__init__()
        self.feature_extractor = resnet32()
        # self.feature_extractor = ResNet()
        self.feature_extractor.fc = \
            nn.Linear(self.feature_extractor.get_in_fc(), feature_size)   # nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_classes, bias=False)

        self.n_classes = n_classes
        self.n_known = 0

        # List containing exemplar_sets
        # Each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        self.exemplar_sets = []

        # Learning method
        self.BCEwithL = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()   #BCE WITH LOGITS
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
        #                             weight_decay=0.00001)
        self.optimizer = optim.SGD(self.parameters(), lr=2.0,
                                  weight_decay=0.00001)

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

        # self.p_ind = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features + n, bias=False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classify(self, x, transform):
        """Classify images by neares-means-of-exemplars

        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        batch_size = x.size(0)

        if self.compute_means:
            print("Computing mean of exemplars...")
            exemplar_means = []
            for P_y in self.exemplar_sets:
                features = []
                # Extract feature for each exemplar in P_y
                for ex in P_y:
                    with torch.no_grad():
                        ex = Variable(ex).cuda()
                        # ex = Variable(transform(Image.fromarray(ex)), volatile=True).cuda()
                        feature = self.feature_extractor(ex.unsqueeze(0))
                        feature = feature.squeeze()
                        feature.data = feature.data / feature.data.norm()  # L2 Normalize
                        features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # L2 Normalize
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            print("Done")

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        feature = self.feature_extractor(x)  # (batch_size, feature_size)
        for i in range(feature.size(0)):  # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        dists = torch.sqrt((feature - means).pow(2).sum(1)).squeeze()  # (batch_size, n_classes)
        # i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
        # _, preds = dists.min(1) prev
        _, preds = dists.min(1)
        # print(preds)
        # print(dists)

        return preds

    def construct_exemplar_set(self, images, m):
        """Construct an exemplar set for image set

        Args:
            images: np.array containing images of a class
        """
        # Compute and cache features for each example
        features = []
        for img in images:
            with torch.no_grad():
                x = Variable(img).cuda()
                # x = Variable(transform(Image.fromarray(x)), volatile=True).cuda()
                feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()
                feature = feature / np.linalg.norm(feature)  # Normalize
                features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

        exemplar_set = []
        exemplar_features = []  # list of Variables of shape (feature_size,)
        p_ind = []
        for k in range(int(m)):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            # i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
            # p_ind.append(ind[i])
            # print(i)
            # for i in images:
            #   print(i)
            # print(len(images))
            exemplar_set.append(images[i])
            exemplar_features.append(features[i])
            """
            print ("Selected example", i)
            print ("|exemplar_mean - class_mean|:")
            print (np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean)))
            #features = np.delete(features, i, axis=0)
            """

        # self.exemplar_sets.append(np.array(exemplar_set))
        # self.p_ind=p_ind
        self.exemplar_sets.append(exemplar_set)

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:int(m)]

    def combine_dataset_with_exemplars(self, dataset):

        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            for i in range(len(P_y)):
                dataset.append([exemplar_images[i], exemplar_labels[i]])
                # i = i + 1
        return dataset

    def update_representation(self, dataset):

        self.compute_means = True
        labels = []
        for image, label in dataset:
            labels.append(label)
        # Increment number of weights in final fc layer
        classes = list(set(labels))
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.increment_classes(len(new_classes))
        self.cuda()
        print("%d new classes" % (len(new_classes)))

        # Form combined training set
        dataset = self.combine_dataset_with_exemplars(dataset)  #IMPORTANT!!!!
        reprdata = []
        indexes = list(range(len(dataset)))
        for i in range(0, len(dataset)):
            reprdata.append([indexes[i], dataset[i][0], dataset[i][1]]) #indici immagine label

        loader = torch.utils.data.DataLoader(reprdata, batch_size=batch_size,
                                             shuffle=True, num_workers=4, drop_last=True)

        # Store network outputs with pre-update parameters
        q = torch.zeros(len(dataset), self.n_classes).cuda()
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # DEVICE = 'cpu'
        print(DEVICE)
        # ind = 0
        for indices, images, labels in loader:
            # !!!!!!!!!!!!try with torch no grad!!!!!!!!!!!!!

            images = Variable(images).cuda()
            # indices = Variable(torch.tensor(np.array(indexes[ind]))).cuda()
            # images = torch.tensor(images).to(DEVICE)
            g = torch.sigmoid(self.forward(images))
            q[indices] = g.data #%%%%%prev%%%%%%%%
            # print("g.size = {}".format(g.size()))
            # print(g)
            # print("q.size = {}".format(q.size()))
            # print(q)
            # q[list(range(ind, ind+batch_size))] = g.data
            # ind = ind + batch_size

        q = Variable(q).cuda()
        print(q)
        # del images, indices
        # Run network training
        optimizer = self.optimizer

        for epoch in range(num_epochs):
            # ind = 0
            iter = 0
            # for i, (indices, images, labels) in enumerate(loader):
            for indices, images, labels in loader:
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                # indices = indices.cuda()
                y_hot = F.one_hot(labels, self.n_classes)
                optimizer.zero_grad()
                g = self.forward(images)

                # Classification loss for new classes
                # loss = self.cls_loss(g, labels)
                # print(g.size())
                # print(labels.size())

                # gnew = [g[:, y] for y in range(self.n_known, self.n_classes)]
                # gnew.transpose(0,1)
                y_hot = y_hot.type_as(g)
                loss = self.BCEwithL(g[...,self.n_classes-10:self.n_classes], y_hot[...,self.n_classes-10:self.n_classes])
                # labels = labels.type_as(g)
                # loss = sum(self.BCEwithL(g[:, y], labels)\
                #                     for y in range(self.n_known, self.n_classes))
                loss = loss / len(range(self.n_known, self.n_classes))

                # Distilation loss for old classes
                if self.n_known > 0:
                    # g = torch.sigmoid(g)

                    q_i = q[indices]  #%%%%%%prev%%%%%%%%%%
                    # q_i = q[list(range(ind, ind + batch_size))]
                    # ind = ind + batch_size
                    # print("q_i: {}".format(q_i))
                    dist_loss = self.BCEwithL(g[...,:self.n_classes-10], q[...,:self.n_classes-10])
                    # dist_loss = sum(self.dist_loss(g[:, y], q_i[:, y]) \
                    #                 for y in range(self.n_known))
                    dist_loss = dist_loss / self.n_known
                    loss += dist_loss

                loss.backward()
                optimizer.step()

                if (iter + 1) % 10 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' \
                          % (epoch + 1, num_epochs, iter + 1, len(dataset) // batch_size, loss.item()))
                iter = iter + 1
