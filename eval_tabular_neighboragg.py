import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.nn import CrossEntropyLoss

import torchvision
from torchvision import transforms

from utility.loss import DirichletLoss
from utility.utils import check_dataset
from classifiers.small_convnet_mnist3digits import SmallConvNetMNIST3Digits
from learners.NeighborAgg import NeighborAgg, TrustScore
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorboardX import SummaryWriter

import pdb

import argparse
import os
import os.path as osp
import json
import shutil
import random

import numpy as np
from sklearn.metrics import classification_report

import cv2
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ucimlr import classification_datasets
from classifiers import tabular_classifier
from sklearn.model_selection import train_test_split
from learners.NeighborAgg_tree import NeighborAgg

def check_manual_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed) # for cpu
    torch.cuda.manual_seed(seed) # for single GPU
    torch.cuda.manual_seed_all(seed) # for all GPUs

    print("Using seed: {seed}".format(seed=seed))




def main(
    gpu_ids,
    dataset,
    feature_dir,
    dataroot,
    seed,
    batch_size,
    eval_batch_size,
    epochs,
    num_workers,
    output_dir,
    cuda,
    tensorboard,
    lr,
    lr_scheduler,
    step_size,
    gamma,
    max_grad_clip,
    max_grad_norm,
    trust_nn,
    neigh_temp, 
    K,
    classifier_name,
    base_ckpt,
    loss_type):


    params = {
        'similarity_T': 1,
        "num_epochs": 200,
        # "trust_model": "SVC",  # ["LR", "RF", "SVC"],
        # "baseline": "NeighborAGG",
        "save_name": "",
        "TS_alpha": 0.0625,
        "val_k": 5,
        "power_degree": 1.28,
        "lr": 0.005,
        "pool_type": "mean",  # ["nearest"]
        "GNN_dim": 18,

        'with_softmax_feature': True,

        "fair_train": True,
        "filtering": "density",  # "density", "none", "uncertainty"

        "graph_model": "GCNNet_3conv",
        # "SGConvNet_1layer", GatedGraphConvNet", "GMMConvNet","GCNNet", "NNConvNet", "CGConvNet", "TransformerConvNet","GMMConvNet"
        "kdtree_by_class": False,  # whether to build K kdtrees or just one kdtree

        "dataset": dataset,

        "slope": 0,
        "is_more_layer": False,

        "writer": None
    }


    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    gpus = gpu_ids.split(',')
    nGPUs = len(gpus)


    # load dataset
    dset = getattr(classification_datasets, dataset)
    dataset = dset("dataset")
    X = dataset[:][0]
    y = dataset[:][1]
    num_class = np.unique(y).shape[0]

    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)


    # load classifier
    if classifier_name == "LR":
        trainer = tabular_classifier.run_logistic
    elif classifier_name == "SVC":
        trainer = tabular_classifier.run_linear_svc
    elif classifier_name == "RF":
        trainer = tabular_classifier.run_random_forest
    elif classifier_name == "KNN":
        trainer = tabular_classifier.run_KNN
    elif classifier_name == "MLP":
        trainer = tabular_classifier.run_MLP


    # load trustworthiness model
    singal = NeighborAgg(params)


    # EVAULATION
    clf, testing_prediction, testing_confidence_max = trainer(X_train, y_train, X_test, y_test)
    _, new_testing_prediction, new_testing_confidence_max = trainer(X_train_all, y_train_all, X_test, y_test)

    split_metrics = {}
    signal.fit(X_train, y_train)
    y_val_pred_prob = clf.predict_proba(X_val)
    signal.fit_learnable(X_val, y_val, y_val_pred_prob)
    trust_score = signal.get_score(X_test, y_test_pred, y_test_pred_prob)


    ## METRICS #####
    metrics = Metrics(metrics_to_use, X_test.shape[0], num_classes)
    metrics.update(testing_prediction, y_test, trust_score)
    scores = metrics.get_scores(split="test")
    print("Final Metrics:", scores)



    print("Evalution done.")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--feature_dir", type=str)
    parser.add_argument("--dataroot", type=str, default="")

    parser.add_argument("--base_ckpt", type=str)
    parser.add_argument("--classifier_name", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--K", type=int)
    parser.add_argument("--loss_type", type=str, default="CE")

    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_ids", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--no_cuda", action="store_false", dest="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_scheduler", type=str)
    parser.add_argument("--step_size", type=int)
    parser.add_argument("--gamma", type=float)

    parser.add_argument("--trust_nn", type=str)
    parser.add_argument("--neigh_temp", type=float)


    parser.add_argument("--max_grad_clip", type=float)
    parser.add_argument("--max_grad_norm", type=float)

    parser.add_argument("--tensorboard", action="store_true", dest="tensorboard")
    #parser.add_argument("--kl_type", type=str, dest="kl_type")

    args = parser.parse_args()


    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    kwargs = vars(args)

    with open(osp.join(args.output_dir, "hparams.json"), 'w') as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    #del kwargs["gpu_id"]

    main(**kwargs)

    print("DONE.")
