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
import tqdm 

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
    neigh_model):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    gpus = gpu_ids.split(',')
    nGPUs = len(gpus)

    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda"
    check_manual_seed(seed)

    # val_dataset, num_classes, C, H, W = check_dataset(dataset, feature_dir, split="train")
    # val_loader = data.DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     drop_last=False,
    # )

    val_dataset, num_classes, C, H, W = check_dataset(dataset, feature_dir, split="train")
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

    from utility import neighboragg_models
    model = getattr(neighboragg_models, neigh_model)
    neigh_agg = model(K*num_classes, num_classes, num_classes, slope=0.2)
    neigh_agg = neigh_agg.to(device)

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(neigh_agg.parameters(), lr=lr)

    if nGPUs > 1:
        f = torch.nn.DataParallel(f, range(nGPUs))
        neigh_agg = torch.nn.DataParallel(neigh_agg, range(nGPUs))

    sav_folder = osp.join(output_dir, "vis_results")
    if not osp.exists(sav_folder): os.makedirs(sav_folder)

    if tensorboard:
        runs_folder = osp.join(output_dir, "neighagg_runs")
        if not osp.exists(runs_folder): os.makedirs(runs_folder)
        writer = SummaryWriter(logdir=runs_folder)

    cnt = 0
    pred_list = []
    neigh_pred_list = []
    score_list = []
    epsilon = 1e-3
    
    ########## VALIDATION ###########
    print("Begin training...")
    logits_list = []
    neighbor_dist_list = []
    ys_list = []
    zs_list = []
    neigh_agg.train()
    loss_list = []
    for epoch in range(epochs):
        for itr, (xs, ys, _, neighbor_dist, _) in enumerate(val_loader):
            optimizer.zero_grad()
            xs = xs.to(device).float() # ---> [B, C, H, W]
            ys = ys.to(device)
            neighbor_dist = neighbor_dist.to(device).float()
            _, logits, zs = f(xs) # zs: [B, Cf, Hf, Wf]
            # softmax_vec = torch.nn.functional.softmax(logits, dim=-1)
            neigh_logits = neigh_agg(neighbor_dist, logits)

            loss = loss_func(neigh_logits, ys)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
            if tensorboard:
                writer.add_scalar("total_loss", loss.item(), itr + epoch * len(val_loader))
            
            print("Epoch {} Iter {}: Loss {}".format(epoch+1, itr+1, loss.item()))
        # if (epoch+1) % 10 == 0:  
    torch.save(neigh_agg.state_dict(), os.path.join(output_dir, "{}_lr{}_train_e{}.pt".format(neigh_model, str(lr).replace(".", ""), epoch+1)))

    plt.plot(range(len(loss_list)), loss_list)
    plt.savefig(osp.join(output_dir,"neighagg_runs/misclf_logits_{}_lr{}_e{}".format(neigh_model, str(lr).replace(".", ""), epoch+1)))
    plt.close()
    # pretrain
    # base_ckpt = os.path.join(output_dir, "{}_lr{}_e{}.pt".format(neigh_model, str(lr).replace(".", ""), epochs))
    # ckpt = torch.load(base_ckpt)
    # neigh_agg.load_state_dict(ckpt)
    # neigh_agg = neigh_agg.to(device)
  
    ############## TEST ###################
    preds = []
    neigh_preds = []
    logits_list = []
    neighbor_dist_list = []
    ys_list = []
    zs_list = []
    neigh_agg.eval()
    for itr, (xs, ys, _, neighbor_dist, _) in enumerate(test_loader):
        with torch.no_grad():
            xs = xs.to(device).float() # ---> [B, C, H, W]
            ys = ys.to(device)
            neighbor_dist = neighbor_dist.to(device).float()
            _, logits, zs = f(xs) # zs: [B, Cf, Hf, Wf]
            # softmax_vec = torch.nn.functional.softmax(logits, dim=-1)
            neigh_logits = neigh_agg(neighbor_dist, logits)

            neigh_score = torch.nn.functional.softmax(neigh_logits, dim=-1)
            score = torch.max(neigh_score, dim=-1)[0]
            pred = (ys == torch.argmax(logits, dim=1)).detach().cpu().numpy()
            neigh_pred = (ys == torch.argmax(neigh_logits, dim=1)).detach().cpu().numpy()
            preds.append(pred)
            neigh_preds.append(neigh_pred)
            score_list.append(score.detach().cpu().numpy())
            # ys_list.append(ys.detach().cpu().numpy())

    preds = np.concatenate(preds)
    scores = np.concatenate(score_list)
    test_acc = np.sum(preds) / preds.shape[0]

    neigh_preds = np.concatenate(neigh_preds)
    neigh_acc = np.sum(neigh_preds) / neigh_preds.shape[0]

    # ys_list = np.concatenate(ys_list)

    ########## METRICS #################
    aucroc = average_precision_score(~preds, -scores)
    from utility.metrics import Metrics
    metrics = Metrics(scores.shape[0], num_classes)
    metrics.update(preds, scores)
    metrics = metrics.get_scores(split="test")

    with open(osp.join(output_dir, "metric.txt"), 'a') as fp:
        # json.dump(scores, fp, sort_keys=True, indent=4)
        fp.write("\nNeighborAgg_train\t")
        for metric_name, metric in metrics.items():
            fp.write(metric['string'] + "\t")
        


    ########### PLOTS ##################
    correct_scores = scores[np.where(preds==True)[0]]
    incor_scores = scores[np.where(preds==False)[0]]
    
    vis_data = np.concatenate([correct_scores, incor_scores], axis=0)
    np.save("cor_incor_scores_neighagg", vis_data)
    plt.hist(incor_scores, bins=100, alpha=0.7, color='red', label="wrongly classified", density=True)
    plt.hist(correct_scores, bins=100, alpha=0.5, color='green', label="correctly classified", density=True)

    plt.legend()
    plt.title("acc:{:.3f}->neigh acc:{:.3f}/auprc:{:.3f}".format(test_acc, neigh_acc, aucroc))
    plt.xlabel("")
    plt.ylabel("neighboragg score")
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(sav_folder, "misclf_logits_{}_lr{}_train_e{}".format(neigh_model, str(lr).replace(".", ""), epochs)))


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
    parser.add_argument("--neigh_model", type=str)

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

    # with open(osp.join(args.output_dir, "neighagg_runs/neighbor.log"), 'a') as fp:
    #     json.dump(kwargs, fp, sort_keys=True, indent=4)

    #del kwargs["gpu_id"]

    main(**kwargs)

    print("DONE.")
