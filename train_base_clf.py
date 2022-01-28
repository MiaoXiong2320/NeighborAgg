import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.nn import CrossEntropyLoss

import torchvision
from torchvision import transforms

from classifiers.small_convnet_mnist3digits import SmallConvNetMNIST3Digits
from classifiers.small_convnet_mnist import SmallConvNetMNIST
import classifiers

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
from utils import check_dataset


def check_manual_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)  # for cpu
    torch.cuda.manual_seed(seed)  # for single GPU
    torch.cuda.manual_seed_all(seed)  # for all GPUs

    print("Using seed: {seed}".format(seed=seed))


def get_learning_rate(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list += [param_group['lr']]
    return lr_list


def main(
        gpu_ids,
        dataset,
        classifier_name,
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
        max_grad_norm):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    gpus = gpu_ids.split(',')
    nGPUs = len(gpus)

    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda"
    check_manual_seed(seed)

    train_dataset, val_dataset, num_classes, C, H, W = check_dataset(dataset)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # f = SmallConvNetMNIST3Digits()
    from classifiers import get_base_classifier
    f = get_base_classifier(classifier_name)

    # classifier_ckpt = torch.load("../classifier/mnist_pretrained/baseline/model_epoch_056.ckpt")
    # f.load_state_dict(classifier_ckpt['model_state_dict'])

    f = f.to(device)
    # f = f.eval()
    ce_loss = CrossEntropyLoss().to(device)

    if nGPUs > 1:
        f = torch.nn.DataParallel(f, range(nGPUs))

    optimizer = optim.Adam([{"params": f.parameters()}],
                           lr=lr,
                           weight_decay=5e-4)  # USE momentum optimizer???

    if lr_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20000,
                                                               verbose=True)  ## should be put into the inner loop

    runs_folder = osp.join(output_dir, "runs")
    if not osp.exists(runs_folder): os.makedirs(runs_folder)
    ckpts_folder = osp.join(output_dir, "ckpts")
    if not osp.exists(ckpts_folder):    os.makedirs(ckpts_folder)

    if tensorboard:
        writer = SummaryWriter(logdir=runs_folder)

    print("Start training...")
    epoch = 0

    for epoch in range(epochs):
        lr_list = get_learning_rate(optimizer)
        print("Learninig rate used: ", lr_list)

        for itr, (xs, ys) in enumerate(train_loader):
            global_step = itr + epoch * len(train_loader)
            optimizer.zero_grad()

            xs = xs.to(device).float()  # ---> [B, C, H, W]
            ys = ys.to(device).long()  # ---> [B, K, C, H, W]

            _, logits, _ = f(xs)

            total_loss = ce_loss(logits, ys)
            total_loss.backward()

            optimizer.step()

            if tensorboard:
                writer.add_scalar("total_loss", total_loss.item(), global_step)

            print("Epoch {} Iter {}: Loss {}".format(epoch + 1, itr + 1, total_loss.item()))

            if lr_scheduler == "ReduceLROnPlateau":
                scheduler.step(total_loss.item())

        # End of each epoch
        print("Saving models...")
        dict2save = {'epoch': epoch,
                     'global_step': global_step,
                     'state_dict': f.state_dict() if nGPUs == 1 else f.module.state_dict()}

        if (epoch + 1) % 1 == 0:
            torch.save(dict2save, osp.join(ckpts_folder, "ckpt_e{}.pt".format(epoch + 1)))

        ##########################################3
        print("Evaluating test set...")
        f.eval()

        preds_list = []
        ys_list = []
        with torch.no_grad():
            for _, (images, labels) in enumerate(val_loader):
                images = images.to(device).float()
                labels = labels.to(device).long()

                _, logits, _ = f(images)
                logps = F.log_softmax(logits, dim=1)
                preds = torch.argmax(logps, dim=1).cpu().detach().clone().numpy()
                ys = labels.cpu().numpy()

                preds_list.append(preds)
                ys_list.append(ys)

        preds = np.concatenate(preds_list, axis=0)
        ys = np.concatenate(ys_list, axis=0)

        # clss_rep = classification_report(ys, preds, output_dict=True)
        test_acc = (preds == ys).sum() / preds.shape[0]

        if tensorboard:
            writer.add_scalar("test acc", test_acc, global_step)

        print("testacc: {}\n".format(test_acc))

        f.train()

        if lr_scheduler == "StepLR":
            scheduler.step()

    print("Training completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--classifier_name", type=str)

    parser.add_argument("--dataroot", type=str, default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_ids", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--no_cuda", action="store_false", dest="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", type=str)
    parser.add_argument("--step_size", type=int)
    parser.add_argument("--gamma", type=float)

    parser.add_argument("--max_grad_clip", type=float)
    parser.add_argument("--max_grad_norm", type=float)

    parser.add_argument("--tensorboard", action="store_true", dest="tensorboard")
    # parser.add_argument("--kl_type", type=str, dest="kl_type")

    args = parser.parse_args()
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    kwargs = vars(args)

    with open(osp.join(args.output_dir, "hparams.json"), 'w') as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    # del kwargs["gpu_id"]

    main(**kwargs)

    print("DONE.")
