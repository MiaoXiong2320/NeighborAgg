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
    graph_model):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    gpus = gpu_ids.split(',')
    nGPUs = len(gpus)

    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda"
    check_manual_seed(seed)


    train_dataset, num_classes, C, H, W = check_dataset(dataset, feature_dir, split="train")
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


    val_dataset, num_classes, C, H, W = check_dataset(dataset, feature_dir, split="val")
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    test_dataset, num_classes, C, H, W = check_dataset(dataset, feature_dir, split="test")
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # K = 100
    from classifiers import get_base_classifier
    f = get_base_classifier(classifier_name)
    classifier_ckpt = torch.load(base_ckpt)
    f.load_state_dict(classifier_ckpt['state_dict'])
    f = f.to(device)
    f = f.eval()
    
    # model = getattr(trust_models, trust_nn)
    # t_nn = model(convf_dim=4608, num_classes=num_classes).to(device)
    # tnn_ckpt = torch.load(os.path.join(output_dir, "ckpts/ckpt_e100.pt"))
    # t_nn.load_state_dict(tnn_ckpt['state_dict'])
    # t_nn.eval()

    # if nGPUs > 1:
    #     f = torch.nn.DataParallel(f, range(nGPUs))
    #     t_nn = torch.nn.DataParallel(t_nn, range(nGPUs))

    sav_folder = osp.join(output_dir, "vis_results")
    if not osp.exists(sav_folder): os.makedirs(sav_folder)

    cnt = 0
    pred_list = []
    neigh_pred_list = []
    score_list = []
    epsilon = 1e-3

    ########## TRAINING ###########
    logits_list = []
    ys_list = []
    neighbor_dist_list = []
    neighbor_indices_list = []
    for itr, (xs, ys, _, neighbor_dist, neighbor_indices) in enumerate(train_loader):
        with torch.no_grad():
            xs = xs.to(device).float() # ---> [B, C, H, W]
            _, logits, zs = f(xs) # zs: [B, Cf, Hf, Wf]
            softmax_vec = torch.nn.functional.softmax(logits, dim=-1)
            # softmax_vec = logits

            neighbor_indices_list.append(neighbor_indices.numpy())
            ys_list.append(ys.numpy())
            neighbor_dist_list.append(neighbor_dist.numpy())
            logits_list.append(softmax_vec.detach().cpu().numpy())

    train_neighbor_indices = np.concatenate(neighbor_indices_list, axis=0)
    train_ys= np.concatenate(ys_list, axis=0)
    train_neigh_dists= np.concatenate(neighbor_dist_list, axis=0)
    train_logits = np.concatenate(logits_list, axis=0)

    train_preds = np.argmax(train_logits, axis=-1)
    train_pred_res = train_preds == train_ys
    train_acc = np.sum(train_pred_res) / train_pred_res.shape[0]


    ########## VALIDATION ###########
    logits_list = []
    ys_list = []
    neighbor_dist_list = []
    neighbor_indices_list = []
    for itr, (xs, ys, _, neighbor_dist, neighbor_indices) in enumerate(val_loader):
        with torch.no_grad():
            xs = xs.to(device).float() # ---> [B, C, H, W]
            _, logits, zs = f(xs) # zs: [B, Cf, Hf, Wf]
            softmax_vec = torch.nn.functional.softmax(logits, dim=-1)
            # softmax_vec = logits

            neighbor_indices_list.append(neighbor_indices.numpy())
            ys_list.append(ys.numpy())
            neighbor_dist_list.append(neighbor_dist.numpy())
            logits_list.append(softmax_vec.detach().cpu().numpy())

    val_neighbor_indices = np.concatenate(neighbor_indices_list, axis=0)
    val_ys= np.concatenate(ys_list, axis=0)
    val_neigh_dists= np.concatenate(neighbor_dist_list, axis=0)
    val_logits = np.concatenate(logits_list, axis=0)

    val_preds = np.argmax(val_logits, axis=-1)
    val_pred_res = val_preds == val_ys
    val_acc = np.sum(val_pred_res) / val_pred_res.shape[0]

    ############## TEST ###################
    logits_list = []
    ys_list = []
    neighbor_dist_list = []
    neighbor_indices_list = []
    for itr, (xs, ys, _, neighbor_dist, neighbor_indices) in enumerate(test_loader):
        with torch.no_grad():
            xs = xs.to(device).float() # ---> [B, C, H, W]
            _, logits, zs = f(xs) # zs: [B, Cf, Hf, Wf]
            softmax_vec = torch.nn.functional.softmax(logits, dim=-1)
            # softmax_vec = logits

            neighbor_indices_list.append(neighbor_indices.numpy())
            ys_list.append(ys.numpy())
            neighbor_dist_list.append(neighbor_dist.numpy())
            logits_list.append(softmax_vec.detach().cpu().numpy())

    test_neighbor_indices = np.concatenate(neighbor_indices_list, axis=0)
    test_ys= np.concatenate(ys_list, axis=0)
    test_neigh_dists= np.concatenate(neighbor_dist_list, axis=0)
    test_logits = np.concatenate(logits_list, axis=0)

    test_preds = np.argmax(test_logits, axis=-1)
    test_pred_res = test_preds == test_ys
    test_acc = np.sum(test_pred_res) / test_pred_res.shape[0]


    from sklearn.metrics import roc_auc_score, average_precision_score
    ######### NEIGHBORKGNN ##########
    args = {"num_epochs": 100, "val_k": K, "filtering": None, "TS_alpha": 0.0625, "kdtree_by_class": True, "similarity_T":1,
            "slope":0.2, "use_relu":True, "is_more_layer":True, "optimizer": "Adam", "lr":0.01, "weight_decay":5e-4, 'graph_model':graph_model, "softmax_node_feature":True, "writer":None, "GNN_dim": 40}
    from learners.NeighborKGNN import NeighborKGNN
    neigh_agg = NeighborKGNN(args)
    neigh_agg.fit_learnable(train_neighbor_indices, train_neigh_dists, train_ys, val_neighbor_indices, val_neigh_dists, val_ys, train_logits, val_logits) 

    # eval neighboragg in test dataset
    scores, y_neigh_preds = neigh_agg.get_score(train_neighbor_indices, train_neigh_dists, train_ys, test_neighbor_indices, test_neigh_dists, train_logits, test_logits)
    # eval acc
    neigh_preds = y_neigh_preds == test_ys
    neigh_acc = np.sum(neigh_preds) / neigh_preds.shape[0]

    ########## METRICS #################
    aucroc = average_precision_score(~test_pred_res, -scores)
    from utility.metrics import Metrics
    metrics = Metrics(scores.shape[0], num_classes)
    metrics.update(test_pred_res, scores)
    metrics = metrics.get_scores(split="test")

    with open(osp.join(output_dir, "metric.txt"), 'a') as fp:
        # json.dump(scores, fp, sort_keys=True, indent=4)
        fp.write("\nNeighborKGNN_{}\t".format(args['graph_model']))
        for metric_name, metric in metrics.items():
            fp.write(metric['string'] + "\t")
        

    ########### PLOTS ##################
    correct_scores = scores[np.where(test_pred_res==True)[0]]
    incor_scores = scores[np.where(test_pred_res==False)[0]]
    
    plt.hist(incor_scores, bins=100, alpha=0.7, color='red', label="wrongly classified", density=True)
    plt.hist(correct_scores, bins=100, alpha=0.5, color='green', label="correctly classified", density=True)

    plt.legend()
    plt.title("acc:{:.3f}->neigh acc:{:.3f}/auprc:{:.3f}".format(test_acc, neigh_acc, aucroc))
    plt.xlabel("")
    plt.ylabel("logDir(x)")
    plt.grid(True)
    plt.show()
    plt.savefig(osp.join(sav_folder, "misclassify_neighborkgnn_{}.jpg".format(graph_model)))


    # ######### TRUST SCORE ##########  
    # trust_model = TrustScore(num_classes)
    # scores = trust_model.get_score(neigh_dists, y_preds)
    # y_neigh_preds = trust_model.get_prediction(neigh_dists)

    # neigh_preds = y_neigh_preds == ys
    # neigh_acc = np.sum(neigh_preds) / preds.shape[0]

    # aucroc = average_precision_score(~preds, -scores)

    # correct_scores = scores[np.where(preds==True)[0]]
    # incor_scores = scores[np.where(preds==False)[0]]
    
    # plt.hist(incor_scores, bins=100, alpha=0.7, color='red', label="wrongly classified", density=True)
    # plt.hist(correct_scores, bins=100, alpha=0.5, color='green', label="correctly classified", density=True)

    # plt.legend()
    # plt.title("acc:{:.3f}->neigh acc:{:.3f}/auprc:{:.3f}".format(acc, neigh_acc, aucroc))
    # plt.xlabel("")
    # plt.ylabel("trust score")
    # plt.grid(True)
    # plt.show()
    # plt.savefig(osp.join(sav_folder, "misclassify_trustscore.jpg"))

    # ######### CALIBRATION ##########
    # from netcal.scaling import TemperatureScaling
    # signal = TemperatureScaling()
    # signal.fit(confidence, ys)
    # calibrated = signal.transform(confidence)
    # confidence = np.max(calibrated, axis=1)

    # correct_confidence = confidence[np.where(preds==True)[0]]
    # incor_confidence = confidence[np.where(preds==False)[0]]
    
    # plt.hist(incor_confidence, bins=100, alpha=0.7, color='red', label="wrongly classified", density=True)
    # plt.hist(correct_confidence, bins=100, alpha=0.5, color='green', label="correctly classified", density=True)

    # aucroc = average_precision_score(~preds, -confidence)
    # plt.legend()
    # plt.title("acc:{:.3f}/auprc:{:.3f}".format(acc, aucroc))
    # plt.xlabel("")
    # plt.ylabel("Confidence")
    # plt.grid(True)
    # plt.show()
    # plt.savefig(osp.join(sav_folder, "misclassify_calibration.jpg"))

    # ########## TCP ###########
    # from TCP import TCP
    # args = {"trust_model":"XGB"}
    # signal =  TCP(args)    
    # signal.fit(zs, ys, confidence)
    # scores = signal.get_score(zs)
    
    # aucroc = average_precision_score(~preds, -scores)

    # correct_scores = scores[np.where(preds==True)[0]]
    # incor_scores = scores[np.where(preds==False)[0]]
    
    # plt.hist(incor_scores, bins=100, alpha=0.7, color='red', label="wrongly classified", density=True)
    # plt.hist(correct_scores, bins=100, alpha=0.5, color='green', label="correctly classified", density=True)

    # plt.legend()
    # plt.title("acc:{:.3f}/auprc:{:.3f}".format(acc, aucroc))
    # plt.xlabel("")
    # plt.ylabel("trust score")
    # plt.grid(True)
    # plt.show()
    # plt.savefig(osp.join(sav_folder, "misclassify_tcp.jpg"))


    # from sklearn.metrics import roc_auc_score, average_precision_score
    # ######### CONIFDENCE BASELINE ##########
    # confidence = np.max(confidence, axis=1)
    # aucroc = average_precision_score(~preds, -confidence)
    # correct_confidence = confidence[np.where(preds==True)[0]]
    # incor_confidence = confidence[np.where(preds==False)[0]]
    
    # plt.hist(incor_confidence, bins=100, alpha=0.7, color='red', label="wrongly classified", density=True)
    # plt.hist(correct_confidence, bins=100, alpha=0.5, color='green', label="correctly classified", density=True)

    # plt.legend()
    # plt.title("acc:{:.3f}/auprc:{:.3f}".format(acc, aucroc))
    # plt.xlabel("")
    # plt.ylabel("Confidence")
    # plt.grid(True)
    # plt.show()
    # plt.savefig(osp.join(sav_folder, "misclassify_confidence.jpg"))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--feature_dir", type=str)
    parser.add_argument("--dataroot", type=str, default="")

    parser.add_argument("--base_ckpt", type=str)
    parser.add_argument("--classifier_name", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--K", type=int)
    parser.add_argument("--graph_model", type=str)

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
