"""
Our main method NeighborAgg.
"""
import numpy as np
import random, torch
import torch.nn.functional as F
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import sys, os
sys.path.append("../")

from methods.neighbor_base import NeighborBase
from models.classification import atten_Classification_act
import nni
import pdb


class NeighborAgg(NeighborBase):

  def __init__(self, args):
    """
    input:
        self.by_class = args['kdtree_by_class'] -> indicate whether to build KDTree by its class, otherwise build a whole KDTree
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
    super(NeighborAgg, self).__init__(args)

    self.object_name = "NeighborAgg"
    print("{} init...".format(self.object_name))

    # self.by_class must be true
    self.by_class = True

    self.num_epochs = args['num_epochs']
    

  def fit_learnable(self, X_val, y_val, y_softmax_vec):
    """train a learnable function on the validation set"""
    trust_vec_val = self.get_neighbor_vec(X_val)  ### ----> h_i's: (N, CK)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # self.emb_size = self.n_labels*self.val_k + self.n_labels
    
    classification = atten_Classification_act(self.n_labels*self.val_k, self.n_labels, self.n_labels, self.args).to(device)
    y_softmax_vec = torch.tensor(y_softmax_vec, dtype=torch.float).to(device) # p_i
    trust_vec_val = torch.tensor(trust_vec_val, dtype=torch.float).to(device) # h_i
    y_val = torch.tensor(y_val, dtype=torch.long).to(device) # y_i
    
    if self.args['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(classification.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
    elif self.args['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(classification.parameters(), lr=self.args['lr'], momentum=0.9, weight_decay=self.args['weight_decay'])

    # optimizer = torch.optim.SGD(classification.parameters(), lr=0.1,
    #                     momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # optimizer = torch.optim.SGD(params, lr=0.7)

    classification.train()
    torch.set_grad_enabled(True) 
    loss_list = []
    ap_errors = []
    for epoch in range(self.num_epochs):
        optimizer.zero_grad()
        # feature = torch.cat([trust_vec_val, y_softmax_vec], dim=1)
        out = classification(trust_vec_val, y_softmax_vec)
        # pdb.set_trace()
        loss = F.nll_loss(out, y_val)
        loss_list.append(loss)
        loss.backward()
        optimizer.step()

    # plt.plot(np.arange(len(loss_list)), ap_errors, color='red')
    # plt.title("{} ap_errors".format(self.graph_model_name))
    # plt.show()
    # # os.makedirs("plt_output/", exist_ok=True)
    # plt.savefig(os.path.join("plt_output/", "{}.png".format(self.graph_model_name+"ap_errorsLR")))
    # plt.close()

    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(len(loss_list)), loss_list)
    # plt.title("{} 2input loss function".format(self.dataset_name))
    # plt.show()
    # plt.savefig("../output/plt_output/LR_2input")
    # plt.close()

    # torch.save(classification.state_dict(), '../output/2inputNN_act_{}.pth'.format(self.dataset_name))
    

    self.classification = classification
    print("{} training done.".format(self.object_name))


  def get_score(self, X, y_pred, y_softmax_vec):
    """
    requires datashape: tensor
    1. use GNN to compute sample's embedding
    2. concatenate this embedding with sample's softmax vector
    3. using classification layer to compute its new softmax output
    4. fetch the given class's corresponding score -> softmax score
    """
    torch.set_grad_enabled(False) 
    if not torch.is_tensor(y_softmax_vec):
      y_softmax_vec = torch.tensor(y_softmax_vec, dtype=torch.float).cuda()

    trust_vec_test = self.get_neighbor_vec(X)
    trust_vec_test = torch.tensor(trust_vec_test, dtype=torch.float).cuda()
    
    # feature = torch.cat([trust_vec_test, y_softmax_vec], dim=1)
    pred = self.classification(trust_vec_test, y_softmax_vec).cpu().numpy()
    pred = np.exp(pred)
    score = pred[range(len(y_pred)), y_pred]
    # try:
    #   assert np.sum(np.exp(pred), axis=1) > 0.9 and np.sum(np.exp(pred), axis=1) < 1.1
    # except:
    #   print(np.sum(np.exp(pred), axis=1))
    #   print("GNN score is not normal")
    return score
  def get_Wp_Wh(self):
    wh = self.classification.fc1.weight.detach().cpu().numpy()
    wp = self.classification.fc2.weight.detach().cpu().numpy()
    return wh, wp

  def get_final_para(self):
    wh = self.classification.fc1.weight.detach().cpu().numpy()
    wp = self.classification.fc2.weight.detach().cpu().numpy()
    w = self.classification.layers.weight.detach().cpu().numpy()
    return wh, wp, w

  def get_prediction(self, X, y_softmax_vec):
    """
    Based on the probability provided by the learnable function, return the class with maximum probability
    """
    pass

  def get_softmax_output(self, X, y_softmax_vec):
    """
    Based on the probability provided by the learnable function, return the class with maximum probability
    """
    pass


if __name__ == '__main__':
  from sklearn import datasets
  from ucimlr import classification_datasets
  from sklearn.model_selection import train_test_split


  # params = {
  #         "similarity_T": 10,
  #         "gpu_ids": 1,
  #         "is_nni": True,
  #         "num_epochs": 800,
  #         "mode": "simple_version_tabular", #["simple_version_image"]
  #         "classifier": "LR", #["LR", "RF", "SVC", "KNN", "MLP"], # "NN"
  #         "trust_model": "XGB", # ["LR", "RF", "SVC"],
  #         "baseline": "TCP", # "ablation", # ["MCP", "Calibration", "SoftmaxOnly", "TrustOnly", "Trust Score", "Trust Score Learnable","Trust Score GNN 2", "Trust Score GNN 3", "Trust Score LR"]
  #         "save_name": "",
  #         "TS_alpha": 0.0625,
  #         "val_k": 10,
  #         "power_degree": 1.28,
  #         "lr": 0.005,

  #         'with_softmax_feature': True,

  #         "filtering": "density", # "density", "none", "uncertainty"

  #         "kdtree_by_class": True, # whether to build K kdtrees or just one kdtree

  #         "ds": "Adult",
  #         'slope': 0,
  #         'is_more_layer': False
  # }

  params = {
          'similarity_T': 10,
          "gpu_ids": 0,
          "is_nni": True,
          "num_epochs": 200,
          "mode": "simple_version_tabular", #["simple_version_image"]
          "classifier": "MLP", #["LR", "RF", "SVC", "KNN", "MLP"], # "NN"
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
          "kdtree_by_class": True, # whether to build K kdtrees or just one kdtree

          "ds":  "Landsat", #"Landsat",

          "split": 1,
          "slope": 0.1,
          "is_more_layer": False,

          # "dataset": ["Iris", "Digits", "Bankruptcy"], 
          "dataset": None, #["Adult", "BankMarketing", "CardDefault", "Landsat", "LetterRecognition", "MagicGamma", "SensorLessDrive", "Shuttle"] # ["Adult", "Avila"
          # Bankruptcy dataset comes from kaggle and needs downloading -> data imbalance, 3410 samples(110 positive)

          "writer": None
  }

  if params['is_nni']:
          import nni
          from nni.utils import merge_parameter


  if params['is_nni']:
          tuner_params = nni.get_next_parameter()
          print(tuner_params)
          params = merge_parameter(params, tuner_params)
          print(params)


  datasets = [params['ds']] #["CardDefault"] 
  for dataset_name in datasets:
    dset = getattr(classification_datasets, dataset_name)
    dataset = dset("dataset")
    X = dataset[:][0]
    y = dataset[:][1]
    num_classes = np.max(y) + 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42)

    from sklearn.linear_model import LogisticRegression
    # Train logistic regression on digits.
    # model = RandomForestClassifier()
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    train_softmax = model.predict_proba(X_train)
    softmax_vac = model.predict_proba(X_val)
    test_softmax = model.predict_proba(X_test)

    # from tensorboardX import SummaryWriter
    # import time
    # # time_stamp = time.strftime("%m%d-%H%M%S", time.localtime())
    # writer = SummaryWriter(log_dir=os.path.join("../output/tb",dataset_name))

    # args = {
    #   "kdtree_by_class":True,
    #   "num_epochs": 800,
    #   "filtering": "density",
    #   "TS_alpha": 0.0625,
    #   "ds": dataset_name,
    #   "writer": None
    # }

    score_model = NeighborAgg(params)
    # from trustscore_LR_2input import TrustScore_LR
    # score_model = TrustScore_LR(params, k=10, alpha=0.0625, filtering="none", min_dist=1e-12)
    score_model.fit(X_train, y_train)
    score_model.fit_learnable(X_val, y_val, softmax_vac)

    wh, wp = score_model.get_Wp_Wh()
    pdb.set_trace()

    # trust_score = score_model.get_score(X_test, model.predict(X_test), model.predict_proba(X_test))

    # import sys, os, sklearn
    # sys.path.append("../")
    # from models.classification import Classification
    # from utils.metrics import Metrics, compute_average_metrics
    # from utils.metric_diff import compute_metric_diff
    # metrics_to_use = ['auc_roc', 'ap_success', 'ap_errors', "fpr_at_95tpr"] # , "fpr_at_95tpr" has bugs!
    # test_pred = model.predict(X_test)
    # metrics = Metrics(metrics_to_use, X_test.shape[0], num_classes)
    # metrics.update(test_pred, y_test, trust_score)
    # scores = metrics.get_scores(split="test")
    # print(scores) 
    # nni.report_final_result(scores['test/ap_errors']['value'])
