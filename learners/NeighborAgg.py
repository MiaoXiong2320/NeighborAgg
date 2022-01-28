import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
import pdb

import random, torch
from torch import nn
import torch.nn.functional as F

class NeighborAgg_NN(nn.Module):
	"""a linear layer for classification
	feat1: trust vector
	feat2: softmax vector
	"""
	def __init__(self, fea1, fea2, num_classes, slope):
		super(NeighborAgg_NN, self).__init__()


		self.slope = slope
		# self.if_act = args['if_act']
		self.fc1 = nn.Linear(fea1+fea2, 400)
		self.fc2 = nn.Linear(400, 400)
		self.fc3 = nn.Linear(400, 400)
		self.fc4 = nn.Linear(400, 400)
		self.fc5 = nn.Linear(400, num_classes)
		self.relu = nn.LeakyReLU(self.slope)

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, feat1, feat2):
		feat = torch.cat((feat1, feat2), dim=1)
		out = self.relu(self.fc1(feat))
		out = self.relu(self.fc2(out))
		out = self.relu(self.fc3(out))
		out = self.relu(self.fc4(out))
		out = self.relu(self.fc5(out))
		# logists = torch.log_softmax(out, 1)
		return out

class NeighborAgg_Dense(nn.Module):
	"""a linear layer for classification
	feat1: trust vector
	feat2: softmax vector
	"""
	def __init__(self, fea1, fea2, num_classes, args):
		super(NeighborAgg_Dense, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.slope = args['slope']
		self.fc1 = nn.Linear(fea1, num_classes)
		self.fc2 = nn.Linear(fea2, num_classes)
		self.relu = nn.LeakyReLU(self.slope)

		self.more_layers = args['is_more_layer']
		if self.more_layers:
			self.layers = nn.Sequential(
				nn.BatchNorm1d(2*num_classes),
				nn.ReLU(),
				nn.Linear(2*num_classes, 8*num_classes),
				nn.BatchNorm1d(8*num_classes),
				nn.ReLU(),
				nn.Linear(8*num_classes, 4*num_classes),
				nn.BatchNorm1d(4*num_classes),
				nn.ReLU(),
				nn.Linear(4*num_classes, num_classes),
				nn.BatchNorm1d(num_classes),
				nn.ReLU(),
			)
		else:
			self.layers = nn.Linear(2*num_classes, num_classes)


		# self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, feat1, feat2):
		if self.use_relu:
			out1 = self.relu(self.fc1(feat1))
			out2 = self.relu(self.fc2(feat2))
		else:
			out1 = self.fc1(feat1)
			out2 = self.fc2(feat2)
		out = torch.cat((out1, out2), dim=1)
		out = self.layers(out)
		# logists = torch.log_softmax(out, 1)
		return out


class atten_Classification_act(nn.Module):
	"""a linear layer for classification
	feat1: trust vector
	feat2: softmax vector
	"""
	def __init__(self, fea1, fea2, num_classes, args):
		super(atten_Classification_act, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.slope = args['slope']
		# self.if_act = args['if_act']
		self.fc1 = nn.Linear(fea1, num_classes)
		self.fc2 = nn.Linear(fea2, num_classes)
		self.relu = nn.LeakyReLU(self.slope)
		self.use_relu = args['use_relu']
		if self.use_relu:
			print("use relu")
		else:
			print("not use relu")

		self.more_layers = args['is_more_layer']
		if self.more_layers:
			self.layers = nn.Sequential(
				nn.BatchNorm1d(2*num_classes),
				nn.ReLU(),
				nn.Linear(2*num_classes, 8*num_classes),
				nn.BatchNorm1d(8*num_classes),
				nn.ReLU(),
				nn.Linear(8*num_classes, 4*num_classes),
				nn.BatchNorm1d(4*num_classes),
				nn.ReLU(),
				nn.Linear(4*num_classes, num_classes),
				nn.BatchNorm1d(num_classes),
				nn.ReLU(),
			)
		else:
			self.layers = nn.Linear(2*num_classes, num_classes)


		# self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, feat1, feat2):
		if self.use_relu:
			out1 = self.relu(self.fc1(feat1))
			out2 = self.relu(self.fc2(feat2))
		else:
			out1 = self.fc1(feat1)
			out2 = self.fc2(feat2)
		out = torch.cat((out1, out2), dim=1)
		out = self.layers(out)
		# logists = torch.log_softmax(out, dim=1)
		return out


class NeighborBase(object):
  def __init__(self, args, min_dist=1e-12):
    """
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
    self.args = args
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
    self.loss = args['loss_type']
    self.classification = None
    

  def fit_learnable(self, y_val, y_softmax_vec, trust_vec_val, test_neigh_dists=None, test_preds=None, test_logits=None, test_pred_res=None):
    """train a learnable function on the validation set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.n_labels = np.max(y_val)+1
    # self.emb_size = self.n_labels*self.val_k + self.n_labels
    
    classification = atten_Classification_act(self.n_labels*self.val_k, self.n_labels, self.n_labels, self.args).to(device)
    self.classification = atten_Classification_act(self.n_labels*self.val_k, self.n_labels, self.n_labels, self.args).to(device)
    y_softmax_vec = torch.tensor(y_softmax_vec, dtype=torch.float).to(device) # p_i
    trust_vec_val = torch.tensor(trust_vec_val, dtype=torch.float).to(device) # h_i
    y_val = torch.tensor(y_val, dtype=torch.long).to(device) # y_i

    #####TEST 
    test_logits = torch.tensor(test_logits, dtype=torch.float).to(device) # p_i
    test_neigh_dists = torch.tensor(test_neigh_dists, dtype=torch.float).to(device) # h_i
    # test_preds = torch.tensor(test_preds, dtype=torch.long).to(device) # y_i
    #####TEST

    if self.args['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(classification.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
    elif self.args['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(classification.parameters(), lr=self.args['lr'], momentum=0.9, weight_decay=self.args['weight_decay'])

    # optimizer = torch.optim.SGD(classification.parameters(), lr=0.1,
    #                     momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # optimizer = torch.optim.SGD(params, lr=0.7)

    if "Focal" in self.loss:
      # from utils import FocalLoss, FocalCE
      # loss_func = FocalLoss(alpha=0.25, gamma=1)
      from utility import utils
      loss_func_name = getattr(utils, self.loss)
      loss_func = loss_func_name()
    elif "Mix" in self.loss:
      from utility.utils import Focal, FocalCE
      focal = Focal()
      ce = FocalCE()



    classification.train()
    torch.set_grad_enabled(True) 
    loss_list = []
    auprc_list = []
    max_auprc=0
    max_epoch = 0
    for epoch in range(self.num_epochs):
        classification.train()
        optimizer.zero_grad()
        # feature = torch.cat([trust_vec_val, y_softmax_vec], dim=1)
        out = classification(trust_vec_val, y_softmax_vec)
        # pdb.set_trace()
        if self.loss == 'CE':
          out = torch.log_softmax(out, dim=1)
          loss = F.nll_loss(out, y_val)
        elif 'Focal' in self.loss:
          misclf_idx = torch.argmax(y_softmax_vec, dim=1)
          loss = loss_func(out, misclf_idx, y_val)
        elif 'Mix' in self.loss:
          misclf_idx = torch.argmax(y_softmax_vec, dim=1)
          loss_focal = focal(out, misclf_idx, y_val)
          # loss_ce = ce(out, misclf_idx, y_val)
          out = torch.log_softmax(out, dim=1)
          cross = F.nll_loss(out, y_val)
          loss =  0.5 * loss_focal + cross
        loss_list.append(loss.item())
        print("Epoch: {}, NeighborAgg loss: {:.4f}".format(epoch, loss.item()))
        loss.backward()
        optimizer.step()
        

        #####TEST
        classification.eval()
        out = classification(test_neigh_dists, test_logits)
        pred = torch.softmax(out, dim=1).detach().cpu().numpy()  
        score = pred[range(len(test_preds)), test_preds]
        from sklearn.metrics import roc_auc_score, average_precision_score
        auprc = roc_auc_score(~test_pred_res, -score)
        print("auprc:{}".format(auprc))
        auprc_list.append(auprc.item())


        if auprc > max_auprc:
          max_auprc = auprc
          max_epoch = epoch
          import copy
          self.classification.load_state_dict(copy.deepcopy(classification.state_dict()))


    # plt.plot(np.arange(len(loss_list)), ap_errors, color='red')
    # plt.title("{} ap_errors".format(self.graph_model_name))
    # plt.show()
    # # os.makedirs("plt_output/", exist_ok=True)
    # plt.savefig(os.path.join("plt_output/", "{}.png".format(self.graph_model_name+"ap_errorsLR")))
    # plt.close()

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(loss_list)), auprc_list, color='red')
    plt.title("ap_errors")
    plt.show()
    # os.makedirs("plt_output/", exist_ok=True)
    plt.savefig("./output/log/NeighborAUPRC_{}".format(self.loss))
    plt.close()

    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.title("NeighborAgg loss function")
    plt.show()
    plt.savefig("./output/log/NeighborAgg_{}".format(self.loss))
    plt.close()

    # torch.save(classification.state_dict(), '../output/2inputNN_act_{}.pth'.format(self.dataset_name))
    
    #####TEST
    # self.classification = classification
    #####TEST
    print("{} training done. Epoch:{}, Max auprc:{}".format(self.object_name, max_epoch, max_auprc))


  def get_score(self, trust_vec_test, y_pred, y_softmax_vec):
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

    trust_vec_test = torch.tensor(trust_vec_test, dtype=torch.float).cuda()
    
    # feature = torch.cat([trust_vec_test, y_softmax_vec], dim=1)
    pred = self.classification(trust_vec_test, y_softmax_vec)
    pred = torch.log_softmax(pred, dim=1).cpu().numpy()
    pred = np.exp(pred)
    score = pred[range(len(y_pred)), y_pred]
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

  def get_prediction(self, trust_vec_test, y_softmax_vec):
    """
    Based on the probability provided by the learnable function, return the class with maximum probability
    """
    torch.set_grad_enabled(False) 
    if not torch.is_tensor(y_softmax_vec):
      y_softmax_vec = torch.tensor(y_softmax_vec, dtype=torch.float).cuda()

    trust_vec_test = torch.tensor(trust_vec_test, dtype=torch.float).cuda()
    
    # feature = torch.cat([trust_vec_test, y_softmax_vec], dim=1)
    pred = self.classification(trust_vec_test, y_softmax_vec).cpu().numpy()
    pred = np.argmax(pred, axis=-1)
    return pred

  def get_logits(self, trust_vec_test, y_softmax_vec):
    """
    Based on the probability provided by the learnable function, return the class with maximum probability
    """
    torch.set_grad_enabled(False) 
    if not torch.is_tensor(y_softmax_vec):
      y_softmax_vec = torch.tensor(y_softmax_vec, dtype=torch.float).cuda()

    trust_vec_test = torch.tensor(trust_vec_test, dtype=torch.float).cuda()
    
    # feature = torch.cat([trust_vec_test, y_softmax_vec], dim=1)
    pred = self.classification(trust_vec_test, y_softmax_vec)
    return pred


class TrustScore:
  """
    Trust Score: a measure of classifier uncertainty based on nearest neighbors.
  """

  def __init__(self, num_classes, k=10, alpha=0.0625, filtering="none", min_dist=1e-12):
    """
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
    self.k = k
    self.filtering = filtering
    self.alpha = alpha
    self.min_dist = min_dist
    self.n_labels = num_classes

  def filter_by_density(self, X):
    """Filter out points with low kNN density.

    Args:
    X: an array of sample points.

    Returns:
    A subset of the array without points in the bottom alpha-fraction of
    original points of kNN density.
    """
    kdtree = KDTree(X)
    knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
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
    neigh = KNeighborsClassifier(n_neighbors=self.k)
    neigh.fit(X, y)
    confidence = neigh.predict_proba(X)
    cutoff = np.percentile(confidence, self.alpha * 100)
    unfiltered_idxs = np.where(confidence >= cutoff)[0]
    return X[unfiltered_idxs, :], y[unfiltered_idxs]

  def fit(self, X, y, classes=None):
    """use training data to build a KD-Tree.

    WARNING: assumes that the labels are 0-indexed (i.e.
    0, 1,..., n_labels-1).

    Args:
    X: an array of sample points.
    y: corresponding labels.
    """
    if classes is not None:
      self.n_labels = classes
    else:
      self.n_labels = np.max(y) + 1
    self.kdtrees = [None] * self.n_labels
    if self.filtering == "uncertainty":
      X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
    # in the training dataset, for every label, build a KD-Tree
    for label in range(self.n_labels):
      if self.filtering == "none":
        X_to_use = X[np.where(y == label)[0]]
        self.kdtrees[label] = KDTree(X_to_use)
      elif self.filtering == "density":
        X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
        self.kdtrees[label] = KDTree(X_to_use)
      elif self.filtering == "uncertainty":
        X_to_use = X_filtered[np.where(y_filtered == label)[0]]
        self.kdtrees[label] = KDTree(X_to_use)

      if len(X_to_use) == 0:
        print(
            "Filtered too much or missing examples from a label! Please lower "
            "alpha or check data.")

  def get_neighbor_vector(self, X):
        
    d = np.tile(None, (X.shape[0], self.n_labels, self.val_k))
    # for every label, compute every test sample's NN?
    for label_idx in range(self.n_labels):
      # choose the distance to nearest neighbor in every class -> @Todo not know why k=2, can set k=1 
      d[:, label_idx, :] = self.kdtrees[label_idx].query(X, k=self.val_k)[0][:, :self.val_k]
    d = d.reshape((d.shape[0], -1))
    if self.temp != 0:
      d = np.exp(-d.astype(float)/self.temp)
    return d
    
  def get_score(self, neigh_vec, y_pred):
    """Compute the trust scores using the training dataset.

    Given a set of points, determines the distance to each class.

    Args:
    X: an array of sample points.
    y_pred: The predicted labels for these points.

    Returns:
    The trust score, which is ratio of distance to closest class that was not
    the predicted class to the distance to the predicted class.
    """
    d = neigh_vec.reshape(neigh_vec.shape[0], self.n_labels, -1)
    d = d[:, :, 0].reshape(neigh_vec.shape[0], self.n_labels)

    # for every test sample, sort its distance to all neighbors(every class we compute one nearest neighbor)
    sorted_d = np.sort(d, axis=1)
    # choose the distance to the nearest neighbor in the groud truth class
    d_to_pred = d[range(d.shape[0]), y_pred]
    # compute trust score 
    d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred,
                                     sorted_d[:, 0], sorted_d[:, 1])
    return d_to_closest_not_pred / (d_to_pred + self.min_dist)

  def get_dist(self, neigh_vec, y_pred):
    """Compute the trust scores using the training dataset.

    Given a set of points, determines the distance to each class.

    Args:
    X: an array of sample points.
    y_pred: The predicted labels for these points.

    Returns:
    The trust score, which is ratio of distance to closest class that was not
    the predicted class to the distance to the predicted class.
    """
    d = neigh_vec.reshape(neigh_vec.shape[0], self.n_labels, -1)
    d = d[:, :, 0].reshape(neigh_vec.shape[0], self.n_labels)

    # for every test sample, sort its distance to all neighbors(every class we compute one nearest neighbor)
    sorted_d = np.sort(d, axis=1)
    # choose the distance to the nearest neighbor in the groud truth class
    d_to_pred = d[range(d.shape[0]), y_pred]
    # compute trust score 

    return d_to_closest_not_pred / (d_to_pred + self.min_dist)    

  def get_prediction(self, neigh_vec):
    d = neigh_vec.reshape(neigh_vec.shape[0], self.n_labels, -1)
    d = d[:, :, 0].reshape(neigh_vec.shape[0], self.n_labels)

    # for every test sample, sort its distance to all neighbors(every class we compute one nearest neighbor)
    prediction = np.argmin(d, axis=1)
    
    return prediction