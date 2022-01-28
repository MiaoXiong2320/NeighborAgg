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
	def __init__(self, fea1, fea2, num_classes, slope):
		super(NeighborAgg_Dense, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.slope = slope
		self.fc1 = nn.Linear(fea1, num_classes)
		self.fc2 = nn.Linear(fea2, num_classes)
		self.relu = nn.LeakyReLU(self.slope)
		self.use_relu = True

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

class NeighborAgg_BN(nn.Module):
	"""a linear layer for classification
	feat1: trust vector
	feat2: softmax vector
	"""
	def __init__(self, fea1, fea2, num_classes, slope):
		super(NeighborAgg_BN, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.slope = slope
		self.fc1 = nn.Linear(fea1+fea2, 8*num_classes)
		self.use_relu = True

		self.layers = nn.Sequential(
            nn.BatchNorm1d(8*num_classes),
            nn.ReLU(),
            nn.Linear(8*num_classes, 8*num_classes),
            nn.BatchNorm1d(8*num_classes),
            nn.ReLU(),
            nn.Linear(8*num_classes, 4*num_classes),
            nn.BatchNorm1d(4*num_classes),
            nn.ReLU(),
            nn.Linear(4*num_classes, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
        )


		# self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, feat1, feat2):
		out = torch.cat((feat1, feat2), dim=1)
		out = self.fc1(out)
		out = self.layers(out)
		# logists = torch.log_softmax(out, 1)
		return out
