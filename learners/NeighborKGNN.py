"""
NeighborGNN base: (KNN way of graph construction)using softmax vector as gnn's node feature, then gnn's embedding is concatenated with a softmax vector, the new feature embedding is fed into a linear layer. 
"""

# system package 
import sys, os
import numpy as np, torch
import torch.nn.functional as F
sys.path.append("../")

import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize


# self-defined
# from methods.neighborgnn_base import NeighborGNNBase
# from models.classification import Classification
from utility import NNConv


# gnn-related
import torch_geometric
from torch_geometric.data import Data

import pdb

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
import pdb

class NeighborBase(object):
  def __init__(self, args, min_dist=1e-12):
    """
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
    self.args = args
    # self.dataset_name = args['ds']
    self.val_k = args['val_k'] # the number of neighbors queried
    self.filter_k = 10 # used for abnormal neighbor filtering
    self.min_dist = min_dist # used for abnormal neighbor filtering
    self.filtering = args['filtering'] # filtering method
    self.alpha = args['TS_alpha'] # percentage of abnormal neighbors filtered

    self.by_class = args['kdtree_by_class']

    self.temp = args['similarity_T']
    self.sim_kernel = (self.temp is not None)
    print("using similarity kernels with temperature {}".format(self.temp))

  def filter_by_density(self, X):
    """Filter out points with low kNN density.
    Args:
    X: an array of sample points.

    Returns:
    A subset of the array without points in the bottom alpha-fraction of
    original points of kNN density.
    """
    kdtree = KDTree(X)
    knn_radii = kdtree.query(X, self.filter_k)[0][:, -1]
    eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
    return X[np.where(knn_radii <= eps)[0], :]

  def filter_by_uncertainty(self, X, y):
    """Filter out points with high label disagreement amongst its kNN neighbors.
    Args:
    X: an array of sample points.

    Returns:
    A subset of the array without points in the bottom alpha-fraction of
    samples with highest disagreement amongst its k nearest neighbors.
    """
    neigh = KNeighborsClassifier(n_neighbors=self.filter_k)
    neigh.fit(X, y)
    confidence = neigh.predict_proba(X)
    confidence = confidence[range(y.shape[0]), y]
    cutoff = np.percentile(confidence, self.alpha * 100)
    unfiltered_idxs = np.where(confidence >= cutoff)[0]
    return X[unfiltered_idxs, :], y[unfiltered_idxs]

  def fit(self, X, y):
    """use training data to build a KD-Tree.

    WARNING: assumes that the labels are 0-indexed (i.e.
    0, 1,..., n_labels-1).

    Args:
    X: an array of sample points.
    y: corresponding labels.
    """
    self.n_labels = np.max(y) + 1

    if self.by_class:
      self.kdtree = [None] * self.n_labels
      if self.filtering == "uncertainty":
        X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
      # in the training dataset, for every label, build a KD-Tree
      for label in range(self.n_labels):
        if self.filtering == "none":
          X_to_use = X[np.where(y == label)[0]]
        elif self.filtering == "density":
          X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
        elif self.filtering == "uncertainty":
          X_to_use = X_filtered[np.where(y_filtered == label)[0]]
        
        if len(X_to_use) == 0:
          print("Filtered too much or missing examples from a label! Please lower alpha or check data.")
        self.kdtree[label] = KDTree(X_to_use)

    else:
      if self.filtering == "none":
        X_to_use = X
      elif self.filtering == "density":
        X_to_use = self.filter_by_density(X)
      elif self.filtering == "uncertainty":
        X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
        X_to_use = X_filtered
      
      if len(X_to_use) == 0:
        print("Filtered too much or missing examples from a label! Please lower alpha or check data.") 
      else:
        ratio = len(X_to_use) / len(X)
        print("{:.2%} has been filtered out.".format(1 - ratio))               
      self.kdtree = KDTree(X_to_use) 

    print("Building KDTree done.")       


  def get_neighbor_vec(self, X):
    if self.by_class:
      d = np.tile(None, (X.shape[0], self.n_labels, self.val_k))
      # for every label, compute every test sample's NN?
      for label_idx in range(self.n_labels):
        try:
          d[:, label_idx, :] = self.kdtree[label_idx].query(X, k=self.val_k)[0][:, :self.val_k]
        except:
          # one class's number of samples is less than self.val_k
          k = self.kdtree[label_idx].data.shape[0]
          d[:, label_idx, :k] = self.kdtree[label_idx].query(X, k=k)[0][:, :k]
          print("one class's number of samples is less than self.val_k")
      d = d.reshape((d.shape[0], -1)).astype(np.float)
    else:
      num_neighbors = self.n_labels * self.val_k
      d = np.tile(None, (X.shape[0], num_neighbors))
      d[:, :] = self.kdtree.query(X, k=num_neighbors)[0][:, :num_neighbors]
      d = d.reshape((d.shape[0], -1)).astype(np.float)

    if self.sim_kernel:
      d = np.exp(-d.astype(np.float)/self.temp)
    return d
    


class NeighborGNNBase(NeighborBase):
  """
    Use a learnable function to replace the fixed trust score.
  """

  def __init__(self, args):
    """
    input:
        self.by_class = args['kdtree_by_class'] -> indicate whether to build KDTree by its class, otherwise build a whole KDTree
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
    super(NeighborGNNBase, self).__init__(args)

    self.graph_model_name = args['graph_model']
    self.softmax_node_feature = args['softmax_node_feature']
    self.num_epochs = args['num_epochs']
    self.writer = args['writer']

  def _build_graph_dataset_knn(self, train_idx, train_dists, y_train, val_idx, val_dists, train_softmax=None, val_softmax=None):
    """
    input:
      train_idx, y_train, .. 's type is numpy 
    1. convert all labels into one-hot encoding (binary labels into 2-column vector)
    2. node feature = one-hot encoding + softmax feature (if self.self.softmax_node_feature=True)
    3. for every node, compute its val_k neighbors -> construct edge_index and edge_weight based on this
    """
    # build a graph dataset: edge_index = [rows,cols], edge_weight, node features

    # one-hot encoding of labels as node feature
    self.n_labels = 10
    if self.n_labels == 2:
      inverse_y_train = np.logical_not(y_train).astype(int)
      train_one_hot = np.stack((y_train, inverse_y_train)).T
    else:
      train_one_hot = label_binarize(y_train, classes=np.arange(self.n_labels))
    val_one_hot = np.zeros((len(val_idx), self.n_labels))
    node_feat = np.concatenate((train_one_hot, val_one_hot), axis=0)

    # whether to use softmax feature as part of the node feature
    if self.softmax_node_feature:
        softmax_feat = np.concatenate((train_softmax, val_softmax), axis=0)
        node_feat = np.concatenate((node_feat, softmax_feat), axis=1)

    node_feat = torch.tensor(node_feat, dtype=torch.float)
    self.node_feature_size = node_feat.shape[1]

    # if self.writer is not None:
    #   class_labels = np.concatenate((y_train, y_val), axis=0)
    #   self.class_labels = torch.tensor(class_labels, dtype=torch.long)

    # for every sample in X(train, val, test dataset) -> compute its K neighbors of every class
    # edge attribute/weight is the distance between these two samples
    # edge index : [2, num_edges], first row is the start point, second row is end point of an edge
    edge_index = []
    rows = []
    cols = []
    edge_attr = []
    # for training data
    row = np.arange(len(train_idx)).repeat(self.val_k*self.n_labels) # repeat every components val_k times and concatenate all of them: [0, 0, 0, 1, 1, 1]
    rows.append(row)
    cols.append(train_idx.flatten())
    edge_attr.append(train_dists.flatten())
    # for validation data
    row = np.arange(len(val_idx)).repeat(self.val_k*self.n_labels) + len(train_idx)
    rows.append(row)
    cols.append(val_idx.flatten())
    edge_attr.append(val_dists.flatten())
    
    # concatenate these two
    rows = np.concatenate(rows).flatten()
    cols = np.concatenate(cols).flatten()
    edge_attr = np.concatenate(edge_attr).flatten() 


    edge_attr = torch.tensor(np.exp((-edge_attr/self.temp)), dtype=torch.float) # [106440,]
    edge_index = torch.tensor([rows,cols], dtype=torch.long) # [2, num_edges]
    # edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index=edge_index, edge_attr=edge_attr)
    # edge_index, edge_attr = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, edge_weight=edge_attr, fill_value=1.0)
    data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_attr.unsqueeze(1)) # [106440, 1]

    print("direction: ", data.is_directed())
    
    # set train, test, val dataset's mask
    train_idx = torch.tensor(np.arange(len(train_idx)), dtype=torch.long)
    val_idx = torch.tensor(np.arange(len(train_idx), len(train_idx) + len(val_idx)), dtype=torch.long)

    data.train_mask = torch.zeros(len(node_feat), dtype=torch.bool)
    data.val_mask = torch.zeros(len(node_feat), dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True

    return data

  def fit_learnable(self, train_idx, train_dists, y_train, val_idx, val_dists, y_val, train_softmax=None, val_softmax=None):
    """train a learnable function on the validation set"""
    # construct graph 
    data = self._build_graph_dataset_knn(train_idx, train_dists, y_train, val_idx, val_dists, y_val, train_softmax, val_softmax)
    self.data = data

    params = {
      "node_features_size": self.node_feature_size,
      "num_classes": self.n_labels,
      "edge_feature_size": 1,
      "log_softmax": True,
      "GNN_dim": self.args['GNN_dim']
    }
    self.emb_size = self.n_labels

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    # optimizer = torch.optim.Adam(graph_model.parameters(), lr=0.01, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(params, lr=0.7)

    model = getattr(NNConv, self.graph_model_name)
    graph_model = model(params).to(device)
    optimizer = torch.optim.Adam(graph_model.parameters(), lr=0.005, weight_decay=5e-4)


    ######################### Train #############################
    graph_model.train()
    torch.set_grad_enabled(True) 
    loss_list = []
    for _ in range(self.num_epochs):
        optimizer.zero_grad()
        out = graph_model(data)
        loss = F.nll_loss(out[self.data.val_mask], y_val)
        loss_list.append(loss)
        loss.backward()
        optimizer.step()

    # plt.plot(np.arange(len(loss_list)), loss_list)
    # plt.title("{} loss function".format(self.graph_model_name))
    # plt.show()
    # # os.makedirs("plt_output/", exist_ok=True)
    # plt.savefig(os.path.join("plt_output/", "{}GNN2.png".format(self.graph_model_name)))
    # plt.close()
    
    # model.eval()
    # embedding = graph_model(data)
    # feature = torch.cat([embedding[data.val_mask], val_softmax], dim=1)
    # _, pred = classification(feature).max(dim=1)
    # correct = int(pred.eq(y_val).sum().item())
    # acc = correct / int(data.val_mask.sum())
    # print('Accuracy: {:.4f}'.format(acc))
    
    self.graph_model = graph_model  
    print("Graph model training done.")

  def get_embedding(self):
    torch.set_grad_enabled(False) 
    embedding = self.graph_model.get_features(self.data)
    self.writer.add_embedding(embedding, metadata=self.class_labels, tag="gnn_feature")

  def get_score(self, train_idx, train_dists, y_train, test_idx, test_dists, train_softmax, test_softmax):
    """
    requires datashape: tensor
    1. use GNN to compute sample's embedding
    2. concatenate this embedding with sample's softmax vector
    3. using classification layer to compute its new softmax output
    4. fetch the given class's corresponding score -> softmax score
    """
    data = self._build_graph_dataset_knn(train_idx, train_dists, y_train, test_idx, test_dists, train_softmax, test_softmax)

    torch.set_grad_enabled(False) 
    if not torch.is_tensor(test_softmax):
      test_softmax = torch.tensor(test_softmax, dtype=torch.float).cuda()

    out = self.graph_model(data) 
    out = out[data.val_mask]
    pred = out.cpu().numpy()
    score = pred[range(len(y_pred)), y_pred]
    return score

  def get_prediction(self, X, y_softmax_vec):
    """
    Based on the probability provided by the learnable function, return the class with maximum probability
    """
    torch.set_grad_enabled(False) 
    if not torch.is_tensor(y_softmax_vec):
      y_softmax_vec = torch.tensor(y_softmax_vec, dtype=torch.float).cuda()

    embedding = self.graph_model(self.data)
    feature = torch.cat([embedding[self.data.test_mask], y_softmax_vec], dim=1)
    trust_scores = self.classification(feature).cpu().numpy()
    prediction = np.argmax(trust_scores, axis=1)
    return prediction

  def get_softmax_output(self, X, y_softmax_vec):
    """
    Based on the probability provided by the learnable function, return the class with maximum probability
    """
    torch.set_grad_enabled(False) 
    if not torch.is_tensor(y_softmax_vec):
      y_softmax_vec = torch.tensor(y_softmax_vec, dtype=torch.float).cuda()

    embedding = self.graph_model(self.data)
    feature = torch.cat([embedding[self.data.test_mask], y_softmax_vec], dim=1)
    trust_scores = self.classification(feature).cpu().numpy()
    return trust_scores



class NeighborKGNN(NeighborGNNBase):
  """
    Use a learnable function to replace the fixed trust score.
  """

  def __init__(self, args):
    """
    input:
        self.by_class = args['kdtree_by_class'] -> indicate whether to build KDTree by its class, otherwise build a whole KDTree
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
    super(NeighborKGNN, self).__init__(args)
    self.object_name = "NeighborKGNN"
    print("{} using {} init...".format(self.object_name, args['graph_model']))


  def fit_learnable(self, train_idx, train_dists, y_train, val_idx, val_dists, y_val, train_softmax=None, val_softmax=None):
    """train a learnable function on the validation set"""
    # construct graph 
    data = self._build_graph_dataset_knn(train_idx, train_dists, y_train, val_idx, val_dists, train_softmax, val_softmax)
    self.data = data

    params = {
      "node_features_size": self.node_feature_size,
      "num_classes": self.n_labels,
      "edge_feature_size": 1,
      "log_softmax": True,
      "GNN_dim": self.args['GNN_dim']
    }
    self.emb_size = self.n_labels

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    # optimizer = torch.optim.Adam(graph_model.parameters(), lr=0.01, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(params, lr=0.7)

    model = getattr(NNConv, self.graph_model_name)
    graph_model = model(params).to(device)
    optimizer = torch.optim.Adam(graph_model.parameters(), lr=0.005, weight_decay=5e-4)


    ######################### Train #############################
    graph_model.train()
    torch.set_grad_enabled(True) 
    loss_list = []
    for _ in range(self.num_epochs):
        optimizer.zero_grad()
        out = graph_model(data)
        loss = F.nll_loss(out[self.data.val_mask], y_val)
        loss_list.append(loss)
        loss.backward()
        optimizer.step()

    # plt.plot(np.arange(len(loss_list)), loss_list)
    # plt.title("{} loss function".format(self.graph_model_name))
    # plt.show()
    # # os.makedirs("plt_output/", exist_ok=True)
    # plt.savefig(os.path.join("plt_output/", "{}GNN2.png".format(self.graph_model_name)))
    # plt.close()
    
    # model.eval()
    # embedding = graph_model(data)
    # feature = torch.cat([embedding[data.val_mask], val_softmax], dim=1)
    # _, pred = classification(feature).max(dim=1)
    # correct = int(pred.eq(y_val).sum().item())
    # acc = correct / int(data.val_mask.sum())
    # print('Accuracy: {:.4f}'.format(acc))
    
    self.graph_model = graph_model  
    print("Graph model training done.")

  def get_embedding(self):
    torch.set_grad_enabled(False) 
    embedding = self.graph_model.get_features(self.data)
    self.writer.add_embedding(embedding, metadata=self.class_labels, tag="gnn_feature")

  def get_score(self, train_idx, train_dists, y_train, test_idx, test_dists, train_softmax, test_softmax):
    """
    requires datashape: tensor
    1. use GNN to compute sample's embedding
    2. concatenate this embedding with sample's softmax vector
    3. using classification layer to compute its new softmax output
    4. fetch the given class's corresponding score -> softmax score
    """
    data = self._build_graph_dataset_knn(train_idx, train_dists, y_train, test_idx, test_dists, train_softmax, test_softmax)

    torch.set_grad_enabled(False) 
    # if not torch.is_tensor(test_softmax):
    #   test_softmax = torch.tensor(test_softmax, dtype=torch.float).cuda()

    y_pred = np.argmax(test_softmax, axis=1)

    data = data.cuda()
    out = self.graph_model(data) 
    out = out[data.val_mask]
    out = torch.nn.functional.softmax(out, dim=-1)
    logits = out.cpu().numpy()

    scores = logits[range(len(y_pred)), y_pred]
    preds = np.argmax(logits, axis=-1)
    return scores, preds

  def get_prediction(self, train_idx, train_dists, y_train, test_idx, test_dists, y_val, train_softmax, test_softmax):
    """
    Based on the probability provided by the learnable function, return the class with maximum probability
    """
    torch.set_grad_enabled(False) 
    if not torch.is_tensor(test_softmax):
      test_softmax = torch.tensor(test_softmax, dtype=torch.float).cuda()

    embedding = self.graph_model(data)
    feature = torch.cat([embedding[self.data.test_mask], y_softmax_vec], dim=1)
    trust_scores = self.classification(feature).cpu().numpy()
    prediction = np.argmax(trust_scores, axis=1)
    return prediction

  def get_softmax_output(self, X, y_softmax_vec):
    """
    Based on the probability provided by the learnable function, return the class with maximum probability
    """
    torch.set_grad_enabled(False) 
    if not torch.is_tensor(y_softmax_vec):
      y_softmax_vec = torch.tensor(y_softmax_vec, dtype=torch.float).cuda()

    embedding = self.graph_model(self.data)
    feature = torch.cat([embedding[self.data.test_mask], y_softmax_vec], dim=1)
    trust_scores = self.classification(feature).cpu().numpy()
    return trust_scores


if __name__ == '__main__':
  from sklearn import datasets
  from sklearn.model_selection import train_test_split
  # digits = datasets.load_digits()
  # X = digits.data
  # y = digits.target

  from sklearn import datasets as sklearn_datasets
  from ucimlr import classification_datasets

  datasets = ["Landsat"] 
  for dataset_name in datasets:
    dset = getattr(classification_datasets, dataset_name)
    dataset = dset("dataset")
    X = dataset[:][0]
    y = dataset[:][1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42)

    from sklearn.linear_model import LogisticRegression
    # Train logistic regression on digits.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    train_softmax = model.predict_proba(X_train)
    softmax_vac = model.predict_proba(X_val)
    test_softmax = model.predict_proba(X_test)

    # from tensorboardX import SummaryWriter
    # import time
    # # time_stamp = time.strftime("%m%d-%H%M%S", time.localtime())
    # writer = SummaryWriter(log_dir=os.path.join("../output/tb",dataset_name))

    gpu_ids = [4]
    devices_str = ",".join([str(_) for _ in gpu_ids])
    print(devices_str)
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str

    params = {
            'similarity_T': 10,
            "gpu_ids": 0,
            "is_nni": True,
            "num_epochs": 200,
            "mode": "simple_version_tabular", #["simple_version_image"]
            "classifier": "MLP", #["LR", "RF", "SVC", "KNN", "MLP"], # "NN"
            "trust_model": "LR",
            "baseline": "NeighborAgg", # "ablation", # ["MCP", "Calibration", "SoftmaxOnly", "TrustOnly", "Trust Score", "Trust Score Learnable","Trust Score GNN 2", "Trust Score GNN 3", "Trust Score LR"]
            "save_name": "",
            "TS_alpha": 0.0625,
            "val_k": 5,
            "power_degree": 1.28,

            "optimizer": "Adam",
            "lr": 0.05,
            "weight_decay": 5e-4,
            "pool_type": "mean", #["nearest"]
            "GNN_dim": 18,

            'with_softmax_feature': True,


            "fair_train":True,
            "filtering": "density", # "density", "none", "uncertainty"

            "graph_model": "GCNNet_3conv", # "SGConvNet_1layer", GatedGraphConvNet", "GMMConvNet","GCNNet", "NNConvNet", "CGConvNet", "TransformerConvNet","GMMConvNet"
            "softmax_node_feature": True,
            "kdtree_by_class": True, # whether to build K kdtrees or just one kdtree

            "ds": "CardDefault",

            "split": 1,
            "slope": 0.1,
            "is_more_layer": False,

            # "dataset": ["Iris", "Digits", "Bankruptcy"], 
            "dataset": None, #["Adult", "BankMarketing", "CardDefault", "Landsat", "LetterRecognition", "MagicGamma", "SensorLessDrive", "Shuttle"] # ["Adult", "Avila"
            # Bankruptcy dataset comes from kaggle and needs downloading -> data imbalance, 3410 samples(110 positive)

            "writer": None
    }

    score_model = NeighborGNNBase(params)
    score_model.fit(X_train, y_train)
    score_model.fit_learnable(X_train, y_train, X_val, y_val, X_test, y_test, train_softmax, softmax_vac, test_softmax)

    result = score_model.get_score(X_test, model.predict(X_test), model.predict_proba(X_test))
    # score_model.get_embedding()