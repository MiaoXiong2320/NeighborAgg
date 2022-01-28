import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils import spectral_norm

class Conv2dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)),
        )

    def forward(self, x):
        return self.net(x)


class SmallConvNetSVHN_SN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.mc_dropout=False
        self.conv1 = Conv2dSame(3, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = Conv2dSame(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = Conv2dSame(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = Conv2dSame(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv5 = Conv2dSame(64, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = Conv2dSame(128, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = spectral_norm(nn.Linear(2048, self.feature_dim))
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = spectral_norm(nn.Linear(self.feature_dim, 10))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv1_bn(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = self.maxpool1(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout1(out)

        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = self.maxpool2(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.dropout2(out)

        out = F.relu(self.conv5(out))
        out = self.conv5_bn(out)
        out = F.relu(self.conv6(out))
        out = self.conv6_bn(out)
        z = self.maxpool3(out)
        if self.mc_dropout:
            out = F.dropout(z, 0.3, training=self.training)
        else:
            out = self.dropout3(z)

        out = out.view(out.size(0), -1)
        feat = F.relu(self.fc1(out))
        if self.mc_dropout:
            out = F.dropout(feat, 0.3, training=self.training)
        else:
            out = self.dropout4(feat)
        out = self.fc2(out)
        return feat, out, z
