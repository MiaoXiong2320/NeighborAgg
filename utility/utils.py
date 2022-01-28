import numpy as np
import torch
import pdb
from datasets import TensorData, TrustMNIST3Digits, TrustMNIST3Digits_NeighAgg
import torch.nn.functional as F

def get_neigh_counts(neighbor_ys, num_classes): # [B, K]
    counts = torch.zeros([neighbor_ys.shape[0], num_classes], dtype=torch.float)
    for c in range(num_classes):
        count = torch.sum(neighbor_ys == c, dim=1)
        counts[:, c] = count 
    return counts # [B, C]


def check_dataset(dataset, feature_dir, split):
    if dataset in ['TensorMNIST3Digits']:
        data_set = TrustMNIST3Digits()(train=True)
        num_classes, C, H, W = 3, 1, 28, 28
    elif dataset in ['TensorMNIST3Digits_OOD']:
        data_set = TrustMNIST3Digits()
        num_classes, C, H, W = 3, 1, 28, 28
    elif dataset == 'TensorMNIST':
        data_dir = "../classifier/code/simple_cls/data/"+dataset
        data_set = TensorData(feature_dir, data_dir, split=split)
        num_classes, C, H, W = 10, 1, 28, 28
    elif dataset.lower() == 'svhn':
        data_dir = "../classifier/code/simple_cls/data/TensorSVHN"
        data_set = TensorData(feature_dir, data_dir, split=split)
        num_classes, C, H, W = 10, 3, 32, 32
    elif dataset.lower() == 'cifar10':
        data_dir = "../classifier/code/simple_cls/data/TensorCIFAR10"
        data_set = TensorData(feature_dir, data_dir, split=split)
        num_classes, C, H, W = 10, 3, 32, 32
    elif dataset.lower() in ['fashionmnist', 'mnist', 'mnist_toy']:
        data_dir = "../classifier/code/simple_cls/data/Tensor"+dataset.upper()
        data_set= TensorData(feature_dir, data_dir, split=split)
        num_classes, C, H, W = 10, 1, 28, 28  
    else:
        raise Exception("Dataset does not exist")
    return data_set, num_classes, C, H, W


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


class SelfConfidMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self, num_classes, device):
        self.nb_classes = num_classes
        self.weighting = 1
        self.device = device
        super().__init__()

    def forward(self, input, target):
        # input = (logits, tcp_conf)
        probs = F.softmax(input[0], dim=1)
        confidence = torch.sigmoid(input[1]).squeeze()
        # Apply optional weighting
        weights = torch.ones_like(target).type(torch.FloatTensor).to(self.device)
        weights[(probs.argmax(dim=1) != target)] *= self.weighting
        labels_hot = one_hot_embedding(target, self.nb_classes).to(self.device)
        loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2
        return torch.mean(loss)

# class FocalLoss(torch.nn.modules.loss._Loss):
#     def __init__(self, alpha=0.25, gamma=5):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, logits, target):
#         # probs = F.softmax(logits, dim=1)
#         CE_loss = F.cross_entropy(logits, target, reduction="none")
#         pt = torch.exp(-CE_loss)
#         loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
#         return torch.mean(loss)

# class FocalLoss(torch.nn.modules.loss._Loss):
#     def __init__(self, alpha=0.25, gamma=5):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, logits, misclf_idx, target):
#         probs = F.softmax(logits, dim=1)
#         pt = probs[range(len(logits)), misclf_idx] / probs[range(len(logits)), target]
#         CE_loss = F.cross_entropy(logits, target, reduction="none")
#         # pt = torch.exp(-CE_loss)
#         # loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
#         loss = self.alpha * (pt) ** self.gamma * CE_loss
#         return torch.mean(loss)

class Focal(torch.nn.modules.loss._Loss):
    def __init__(self, alpha=1, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, misclf_idx, target):
        probs = F.softmax(logits, dim=1)
        pt = probs[range(len(logits)), misclf_idx] * (misclf_idx == target)
        # pt = probs[range(len(logits)), misclf_idx] / probs[range(len(logits)), target]
        CE_loss = F.cross_entropy(logits, target, reduction="none")
        # pt = torch.exp(-CE_loss)
        # loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        loss = (self.alpha * (1 - pt) ** self.gamma * 0.5 + 1) * CE_loss
        return torch.mean(loss)


class FocalCE(torch.nn.modules.loss._Loss):
    def __init__(self, alpha=0.25, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, misclf_idx, target):
        # probs = F.softmax(logits, dim=1)
        # pt = probs[range(len(logits)), misclf_idx] * (misclf_idx == target)
        # pt = probs[range(len(logits)), misclf_idx] / probs[range(len(logits)), target]
        logits = F.softplus(logits, beta=1, threshold=5)
        CE_loss = F.cross_entropy(logits, misclf_idx, reduction="none")
        # pt = torch.exp(-CE_loss)
        # loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        loss =  CE_loss * ((misclf_idx == target) * 2 -1)
        return torch.mean(loss)

if __name__ == '__main__':
    ys = torch.tensor([[1,1,2,0,2,1],[1,1,2,0,2,1]], dtype=torch.int)
    counts = get_neigh_counts(ys, 3)
 