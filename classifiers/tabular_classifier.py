# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np
import torch
from torch import nn

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.metrics import precision_recall_curve
# import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from utils.plot import plot_precision_curve
import pdb

from models.tabular_NN import MLPModel

import copy


def run_MLP(X_train, y_train, X_test, y_test, get_training=False, lr=0.01, num_epochs=50):
    """Run a NN with a single layer on some data.

    Returns the predicted values as well as the confidences.
    """
    torch.set_grad_enabled(True)
    n_classes = np.max(y_train) + 1
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1000, shuffle=False)

    input_feature_size = X_train.shape[1]
    model = MLPModel(input_feature_size, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()

    is_train = True
    best_acc = 0
    if is_train:
        loss_list = []
        for epoch in range(num_epochs):
            train_loss = 0
            model.train()
            for batch_id, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                output = model(x)

                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss
            loss_list.append(train_loss / batch_id)

            correct = 0
            total = 0
            with torch.no_grad():
                model.eval()
                for batch_id, (x, y) in enumerate(val_loader):
                    x, y = x.cuda(), y.cuda()
                    output = model(x)

                    _, predicted = output.max(1)
                    total += x.size(0)
                    correct += predicted.eq(y).sum().item()

                acc = 100. * correct / total
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model.state_dict())
            print("Epoch: {}, LOSS: {}, ACC: {}".format(epoch, train_loss / batch_id, acc))

        # plt.plot(range(len(loss_list)), loss_list, label='Training loss')
        # plt.legend()
        # plt.show()
        # os.makedirs("output", exist_ok=True)
        # plt.savefig("output/loss.png")
        # plt.close()

        # torch.save(best_model.state_dict(), 'output/tabular_MLP_{}.pth'.format())
    else:
        state = torch.load('output/CNN.pth', map_location='cuda')
        model.load_state_dict(state)

    model.load_state_dict(best_model)
    # direct way
    with torch.no_grad():
        all_confidence = model.predict_proba(X_test)
        y_pred = np.argmax(all_confidence, axis=1)
        confidences = all_confidence[range(len(y_pred)), y_pred]
    result = 1.0 * np.sum(y_pred == y_test.cpu().numpy()) / len(y_pred)
    print("Test Result: ", result)

    print("MLP training done.")
    if not get_training:
        return model, y_pred, confidences


def run_fashionmnist(train_loader, val_loader, test_loader, dataset_name,
                     num_epochs=25,
                     learning_rate=0.01,
                     batch_size=64,
                     get_training=False):
    """Run a NN with a single layer on some data.

    Returns the predicted values as well as the confidences.
    """
    # from models.MNIST import MLPNet
    # model = MLPNet()
    from models.resnet1 import ResNet18
    model = ResNet18()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        # criterion = criterion.cuda()

    is_train = True
    if is_train:
        loss_list = []
        num_epochs = 200
        for epoch in range(num_epochs):
            train_loss = 0
            for batch_id, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                output = model(x)

                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss

            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            print("Epoch: {}, LOSS: {}, ACC: {}".format(epoch, train_loss / batch_id, acc))
            loss_list.append(train_loss / batch_id)

        plt.plot(range(len(loss_list)), loss_list, label='Training loss')
        plt.legend()
        plt.show()
        plt.savefig("output/loss.png")
        plt.close()

        torch.save(model.state_dict(), 'output/CNN_{}.pth'.format(dataset_name))
        print("Training done.")
    else:
        state = torch.load('output/CNN_{}.pth'.format(dataset_name), map_location='cuda')
        model.load_state_dict(state)

    y_pred = []
    confidences = []
    ground_truth = []
    with torch.no_grad():
        for x, y in test_loader:
            ground_truth.append(y)
            x = x.cuda()
            logits = model.predict_proba(x)
            prediction = torch.argmax(logits, dim=1)
            confidence = torch.max(logits, dim=1)[0]
            y_pred.append(prediction)
            confidences.append(confidence)

    y_pred = torch.cat(y_pred).cpu().numpy()
    confidences = torch.cat(confidences).cpu().numpy()
    ground_truth = torch.cat(ground_truth).numpy()

    if not get_training:
        return model, y_pred, confidences, ground_truth


def run_cifar10(train_loader, val_loader, test_loader, dataset_name,
                num_epochs=25,
                learning_rate=0.01,
                batch_size=64,
                get_training=False):
    """Run a NN with a single layer on some data.

    Returns the predicted values as well as the confidences.
    """
    from models.resnet import ResNet18
    model = ResNet18()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
    #                     weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma= 0.1)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        # criterion = criterion.cuda()

    is_train = False
    if is_train:
        loss_list = []
        num_epochs = 200
        for epoch in range(num_epochs):
            train_loss = 0
            for batch_id, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                output = model(x)

                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss

            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            scheduler.step()

            print("Epoch: {}, LOSS: {}, ACC: {}".format(epoch, train_loss / batch_id, acc))
            loss_list.append(train_loss / batch_id)

        plt.plot(range(len(loss_list)), loss_list, label='Training loss')
        plt.legend()
        plt.show()
        plt.savefig("output/loss_cifar10.png")
        plt.close()

        torch.save(model.state_dict(), 'output/resnet18_{}.pth'.format(dataset_name))
        print("Training done.")
    else:
        state = torch.load('output/CNN_{}.pth'.format(dataset_name), map_location='cuda')
        model.load_state_dict(state)

    y_pred = []
    confidences = []
    ground_truth = []
    with torch.no_grad():
        for x, y in test_loader:
            ground_truth.append(y)
            x = x.cuda()
            logits = model.predict_proba(x)
            prediction = torch.argmax(logits, dim=1)
            confidence = torch.max(logits, dim=1)[0]
            y_pred.append(prediction)
            confidences.append(confidence)

    y_pred = torch.cat(y_pred).cpu().numpy()
    confidences = torch.cat(confidences).cpu().numpy()
    ground_truth = torch.cat(ground_truth).numpy()

    if not get_training:
        return model, y_pred, confidences, ground_truth



def run_LeNet(train_loader, test_loader, dataset_name,
              num_epochs=25,
              learning_rate=0.01,
              batch_size=64,
              get_training=False):
    """Run a NN with a single layer on some data.

    Returns the predicted values as well as the confidences.
    """
    from models.LeNet import LeNet5
    model = LeNet5()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        # criterion = criterion.cuda()

    is_train = True
    if is_train:
        loss_list = []
        num_epochs = 25
        for epoch in range(num_epochs):
            train_loss = 0
            for batch_id, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                output = model(x)

                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss
            print("Epoch: {}, LOSS: {}".format(epoch, train_loss / batch_id))
            loss_list.append(train_loss / batch_id)

        plt.plot(range(len(loss_list)), loss_list, label='Training loss')
        plt.legend()
        plt.show()
        plt.savefig("output/loss.png")
        plt.close()

        torch.save(model.state_dict(), 'output/CNN_{}.pth'.format(dataset_name))
        print("Training done.")
    else:
        state = torch.load('output/CNN_{}.pth'.format(dataset_name), map_location='cuda')
        model.load_state_dict(state)

    y_pred = []
    confidences = []
    ground_truth = []
    with torch.no_grad():
        for x, y in test_loader:
            ground_truth.append(y)
            x = x.cuda()
            logits = model.predict_proba(x)
            prediction = torch.argmax(logits, dim=1)
            confidence = torch.max(logits, dim=1)[0]
            y_pred.append(prediction)
            confidences.append(confidence)

    y_pred = torch.cat(y_pred).cpu().numpy()
    confidences = torch.cat(confidences).cpu().numpy()
    ground_truth = torch.cat(ground_truth).numpy()

    if not get_training:
        return model, y_pred, confidences, ground_truth



def run_KNN(X_train, y_train, X_test, y_test, get_training=False):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # pdb.set_trace()
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    if not get_training:
        return model, y_pred, confidences
    y_pred_training = model.predict(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                  y_pred_training]
    return y_pred, confidences, y_pred_training, confidence_training


def run_MLP_sklearn(X_train, y_train, X_test, y_test, get_training=False):
    in_channel = X_train.shape[0]
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(in_channel, in_channel * 2, in_channel * 4), max_iter=300)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    if not get_training:
        return model, y_pred, confidences
    y_pred_training = model.predict(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                  y_pred_training]
    return y_pred, confidences, y_pred_training, confidence_training


def run_logistic(X_train, y_train, X_test, y_test, get_training=False):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    if not get_training:
        return model, y_pred, confidences
    y_pred_training = model.predict(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                  y_pred_training]
    return y_pred, confidences, y_pred_training, confidence_training


def run_linear_svc(X_train, y_train, X_test, y_test, get_training=False):
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # all_confidence = model.decision_function(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    if not get_training:
        return model, y_pred, confidences
    y_pred_training = model.predict(X_train)
    # all_confidence_training = model.decision_function(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                  y_pred_training]
    return y_pred, confidences, y_pred_training, confidence_training


def run_random_forest(X_train, y_train, X_test, y_test, get_training=False):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    if not get_training:
        return model, y_pred, confidences
    y_pred_training = model.predict(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                  y_pred_training]
    return y_pred, confidences, y_pred_training, confidence_training


def get_stderr(L):
    # compute standard error
    return np.std(L) / np.sqrt(len(L))


def accuracy(y_pred, y_test):
    target_points = np.where(y_pred == y_test)[0]
    acc = len(target_points) / (1. * len(y_test))
    return acc

