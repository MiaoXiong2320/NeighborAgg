import torch.nn as nn
import torch.nn.functional as F

#from models.model import AbstractModel
import torch
import pdb


class SmallConvNetMNIST(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        z = self.maxpool(out) ### ---> [64, 12, 12]

        out = self.dropout1(z)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        
        feat = self.dropout2(out) # out =

        out = self.fc2(feat) # out = self.fc2(out)
        return feat, out, z


if __name__  == "__main__":

    classifier = SmallConvNetMNIST()

    ckpt = torch.load("../../classifier/mnist_pretrained/baseline/model_epoch_056.ckpt")
    classifier.load_state_dict(ckpt['model_state_dict'])

    print("DONE.")
